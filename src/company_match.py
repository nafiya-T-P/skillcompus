"""Company job match (extension of the skill-gap analysis).

Reads ONE public job-posting page (or a pasted job description), finds the skills it asks for with the same skill
dictionary used for resumes, and compares them with the resume skills the user has reviewed.

Safety rules for reading a web address typed by a user:
  * only public http/https addresses (no localhost, private networks or cloud metadata addresses) - every redirect is re-checked
  * only ports 80/443, short timeouts, at most 2 MB, at most 3 redirects, HTML pages only
  * robots.txt is respected; login pages and sites that block automated access are reported, never bypassed
Known limit: the address is checked before the request and again for every redirect, but a very determined DNS
trick between check and request is not fully excluded. Do not expose this feature publicly without rate limiting.
"""
from __future__ import annotations

import html as html_lib
import ipaddress
import json
import re
import socket
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

from . import resume_parser, skill_analyzer

MAX_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 3
MAX_TEXT_CHARS = 30_000
MIN_TEXT_CHARS = 200
MIN_JOB_SKILLS = 3
USER_AGENT = "SkillCompassBot/1.0 (student project; reads one public page when a user asks)"
PASTE_HINT = "You can paste the job description text instead."

_resolve = socket.getaddrinfo      # indirection so tests can replace DNS


# ───────────────────────── safe fetching ─────────────────────────
def validate_public_url(url: str) -> str:
    """Return a cleaned URL, or raise ValueError. Only public http(s) addresses are allowed."""
    url = (url or "").strip()
    if not url or len(url) > 2000:
        raise ValueError("Please enter a valid web address (https://...).")
    if "://" not in url:
        url = "https://" + url
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("Only http and https web addresses are supported.")
    if parts.username or parts.password:
        raise ValueError("Web addresses that contain a user name or password are not allowed.")
    try:
        port = parts.port
    except ValueError:
        raise ValueError("Please enter a valid web address (https://...).")
    if port not in (None, 80, 443):
        raise ValueError("Only the standard web ports (80 and 443) are allowed.")
    try:
        infos = _resolve(parts.hostname, port or (443 if parts.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError, OSError):
        raise ValueError("That web address could not be found. Check the spelling.")
    if not infos:
        raise ValueError("That web address could not be found. Check the spelling.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global:
            raise ValueError("That address points to a private or internal network and is not allowed.")
    return url


def _http_get(url: str) -> tuple[int, dict, bytes]:
    """One GET request without following redirects. Returns (status, lower-case headers, body)."""
    try:
        import requests
    except ImportError:
        raise ValueError("The 'requests' package is not installed. Run: pip install requests beautifulsoup4")
    try:
        with requests.get(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
                          timeout=(5, 10), allow_redirects=False, stream=True) as r:
            body = b""
            for chunk in r.iter_content(65536):
                body += chunk
                if len(body) > MAX_BYTES:
                    raise ValueError("The page is too large to read. " + PASTE_HINT)
            return r.status_code, {k.lower(): v for k, v in r.headers.items()}, body
    except requests.RequestException as err:
        raise ValueError(f"The page could not be loaded ({type(err).__name__}). Try again later. " + PASTE_HINT)


def _robots_allows(url: str) -> bool:
    p = urlparse(url)
    try:
        status, _, body = _http_get(validate_public_url(f"{p.scheme}://{p.netloc}/robots.txt"))
    except ValueError:
        return True                      # robots.txt could not be read: treat as "no rules"
    if status >= 500:
        return False                     # RFC 9309: a server error on robots.txt means "do not crawl"
    if status != 200:
        return True
    rp = RobotFileParser()
    rp.parse(body.decode("utf-8", errors="replace").splitlines())
    return rp.can_fetch(USER_AGENT, url)


def fetch_job_page(url: str) -> dict:
    """Read one public job-posting page. Returns {url, title, company, text, source}. Raises ValueError with a user-friendly message."""
    current = validate_public_url(url)
    status, headers, body = 0, {}, b""
    for _ in range(MAX_REDIRECTS + 1):
        if not _robots_allows(current):
            raise ValueError("This website does not allow automated reading of its pages (robots.txt). " + PASTE_HINT)
        status, headers, body = _http_get(current)
        if status in (301, 302, 303, 307, 308) and headers.get("location"):
            current = validate_public_url(urljoin(current, headers["location"]))
            continue
        break
    else:
        raise ValueError("The link redirects too many times.")

    if status in (401, 403, 429, 451):
        raise ValueError("The website blocks automated access or needs a login. " + PASTE_HINT)
    if status in (404, 410):
        raise ValueError("That page was not found. Check the link.")
    if status >= 400:
        raise ValueError(f"The website returned an error ({status}). " + PASTE_HINT)
    content_type = headers.get("content-type", "")
    if content_type and "html" not in content_type.lower():
        raise ValueError("That link is not a web page (it looks like a file or download). " + PASTE_HINT)

    m = re.search(r"charset=([\w-]+)", content_type, re.I)
    try:
        page = body.decode(m.group(1) if m else "utf-8", errors="replace")
    except LookupError:
        page = body.decode("utf-8", errors="replace")
    info = extract_job_info(page)
    if len(info["text"]) < MIN_TEXT_CHARS:
        raise ValueError("Very little readable text was found. The page may need JavaScript or a login. " + PASTE_HINT)
    info["url"] = current
    return info


# ───────────────────────── reading the page ─────────────────────────
def _to_text(value) -> str:
    """Turn a schema.org field (text, list or small object) into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        value = html_lib.unescape(value)
        if "<" in value:
            from bs4 import BeautifulSoup
            value = BeautifulSoup(value, "html.parser").get_text("\n")
        return re.sub(r"[ \t]+", " ", value).strip()
    if isinstance(value, list):
        return ", ".join(t for t in (_to_text(v) for v in value) if t)
    if isinstance(value, dict):
        if "monthsOfExperience" in value:
            try:
                return f"{int(float(value['monthsOfExperience']) // 12)}+ years of experience"
            except (TypeError, ValueError):
                return ""
        return _to_text(value.get("name") or value.get("description") or "")
    return str(value)


def _find_job_posting(soup) -> dict | None:
    """Look for schema.org JobPosting data (JSON-LD) that many career pages include."""
    def walk(node):
        if isinstance(node, list):
            for item in node:
                found = walk(item)
                if found:
                    return found
        elif isinstance(node, dict):
            kind = node.get("@type")
            if kind == "JobPosting" or (isinstance(kind, list) and "JobPosting" in kind):
                return node
            return walk(node.get("@graph")) if "@graph" in node else None
        return None

    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or tag.get_text() or "")
        except (ValueError, TypeError):
            continue
        job = walk(data)
        if job:
            return job
    return None


def extract_job_info(page_html: str) -> dict:
    """Plain job text from a page: structured JobPosting data when present, otherwise the visible main text."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(page_html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""

    job = _find_job_posting(soup)
    if job:
        parts = [_to_text(job.get(k)) for k in ("title", "description", "responsibilities", "qualifications", "skills",
                                                 "experienceRequirements", "educationRequirements")]
        text = "\n".join(p for p in parts if p)
        org = job.get("hiringOrganization")
        company = _to_text(org.get("name") if isinstance(org, dict) else org)
        if len(text) >= MIN_TEXT_CHARS:
            return {"title": _to_text(job.get("title")) or title, "company": company,
                    "text": text[:MAX_TEXT_CHARS], "source": "structured job data on the page"}

    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"]):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = re.sub(r"\n\s*\n+", "\n", root.get_text("\n")).strip()
    return {"title": title, "company": "", "text": text[:MAX_TEXT_CHARS], "source": "visible page text"}


# ───────────────────────── comparing with the resume ─────────────────────────
_EDU_PATTERNS = {"Bachelor's degree": r"\bbachelor|\bb\.?\s?tech\b|\bb\.?e\b|\bbca\b|\bb\.?sc\b",
                 "Master's degree": r"\bmaster|\bm\.?\s?tech\b|\bmca\b|\bm\.?sc\b|\bmba\b",
                 "PhD": r"\bph\.?d\b|\bdoctorate\b"}


def extract_requirements(text: str) -> dict:
    """Experience-years phrases (for example '3+ years') and education levels mentioned in the posting."""
    years: list[str] = []
    for sentence in re.split(r"[.\n;]", text):
        if "experience" not in sentence.lower():
            continue
        for m in re.finditer(r"(\d{1,2})(?:\s*(?:-|–|to)\s*(\d{1,2}))?\s*(\+)?\s*years?", sentence, re.I):
            phrase = f"{m.group(1)}-{m.group(2)} years" if m.group(2) else f"{m.group(1)}{'+' if m.group(3) else ''} years"
            if phrase not in years:
                years.append(phrase)
    education = [label for label, pat in _EDU_PATTERNS.items() if re.search(pat, text, re.I)]
    return {"years": years[:3], "education": education}


def count_mentions(norm_text: str, skill: str) -> int:
    norm_skill = resume_parser.normalize_text(skill).strip()
    return len(re.findall(rf"(?<![\w+#-]){re.escape(norm_skill)}(?![\w+#-])(?!\.\w)", norm_text, re.I))


def analyze_company_fit(resume_skills: list[str], job_text: str, dict_data: dict | None = None) -> dict:
    """Compare reviewed resume skills with the skills a job posting asks for."""
    dict_data = dict_data or resume_parser.load_skills_dictionary()
    job_skills = resume_parser.match_skills(job_text, dict_data)
    if len(job_skills) < MIN_JOB_SKILLS:
        return {"sufficient_data": False,
                "message": f"Only {len(job_skills)} known skill(s) were found in this text, which is too few for a reliable "
                           "comparison. Use a link to a specific job posting, or paste the full job description."}

    gap = skill_analyzer.compute_skill_gap(resume_skills, job_skills, "this job", dict_data.get("learning_suggestions"))
    norm_job = resume_parser.normalize_text(job_text)
    mentions = {s: count_mentions(norm_job, s) for s in job_skills}
    missing = sorted(gap["missing"], key=lambda s: (-mentions.get(s, 0), s.lower()))
    req = extract_requirements(job_text)

    tips = []
    if gap["score"] >= 70:
        tips.append("Your skills cover most of what this posting asks for. Focus on showing them with results, such as numbers, projects and outcomes.")
    for s in missing[:5]:
        n = mentions.get(s, 0)
        tips.append(f"“{s}” appears {n} time{'s' if n != 1 else ''} in the posting. If you have really used it, add it to your Skills "
                    "section and to a project or job bullet. If not, consider learning it.")
    if req["years"]:
        tips.append(f"The posting asks for {', '.join(req['years'])} of experience. State your experience duration clearly near the top of your resume.")
    if req["education"]:
        tips.append(f"The posting mentions {', '.join(req['education'])}. Make sure your education section is easy to find.")
    tips.append("Only add skills you genuinely have. These are suggestions based on one posting, not hiring guarantees.")

    return {"sufficient_data": True, "job_skills": job_skills, "score": gap["score"], "formula": gap["formula"],
            "matched": gap["matched"], "missing": missing, "mentions": mentions, "years": req["years"],
            "education": req["education"], "tips": tips, "learning_suggestions": gap["learning_suggestions"],
            "disclaimer": "Based on a single job posting. A good match does not guarantee an interview or a job offer."}
