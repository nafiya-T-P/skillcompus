"""Company job match: address safety, page reading, fetching rules and the resume-versus-job comparison.
No real website is contacted: DNS and HTTP are replaced by test doubles."""
import socket

import pytest

from src import company_match as cm

DICT = {"skills": ["Python", "SQL", "Docker", "AWS", "Machine Learning", "Tableau", "Spark", "CI/CD"],
        "aliases": {"ml": "Machine Learning"}}

JOB_TEXT = ("We are hiring a Data Engineer. You will build pipelines with Python and SQL on AWS. "
            "Experience with Docker and CI/CD is required. Spark is a plus. SQL skills are essential. "
            "We ask for 3+ years of experience and a Bachelor's degree. Python is used every day.") * 2

LD_JSON_PAGE = """<html><head><title>Careers</title>
<script type="application/ld+json">{"@context":"https://schema.org","@type":"JobPosting","title":"Data Engineer",
"hiringOrganization":{"@type":"Organization","name":"Acme Corp"},
"description":"&lt;p&gt;Build pipelines with Python and SQL on AWS. Docker experience required. We offer a great team, flexible hours and learning budget for everyone who joins us in Bengaluru.&lt;/p&gt;",
"experienceRequirements":{"@type":"OccupationalExperienceRequirements","monthsOfExperience":36}}</script></head>
<body><nav>Home About Contact</nav><main>unrelated text</main></body></html>"""

PLAIN_PAGE = """<html><head><title>Data Analyst - Acme</title><style>.x{color:red}</style></head><body>
<nav>MENU-LINK Careers Blog</nav><header>HEADER-TEXT</header>
<main><h1>Data Analyst</h1><p>You will analyse data with SQL and Tableau and report to managers every week. We expect Python knowledge,
clear communication and the ability to work with several teams across the company on many different projects at the same time.</p></main>
<footer>FOOTER-TEXT Privacy</footer><script>var secret = 1;</script></body></html>"""


def public_dns(host, port, type=None, **kw):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]


def private_dns(host, port, type=None, **kw):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", port))]


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch):
    monkeypatch.setattr(cm, "_resolve", public_dns)


# ───────────────────────── address safety ─────────────────────────
@pytest.mark.parametrize("url", ["", "   ", "file:///etc/passwd", "ftp://example.com/x", "javascript:alert(1)",
                                 "https://user:pass@example.com/", "https://example.com:6379/", "x" * 2100])
def test_bad_addresses_are_rejected(url):
    with pytest.raises(ValueError):
        cm.validate_public_url(url)


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.1.2.3", "192.168.0.10", "172.16.0.1", "169.254.169.254", "::1", "0.0.0.0"])
def test_private_and_internal_addresses_are_rejected(monkeypatch, ip):
    monkeypatch.setattr(cm, "_resolve", lambda host, port, type=None, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))])
    with pytest.raises(ValueError, match="private or internal"):
        cm.validate_public_url("https://some-site.example/jobs")


def test_public_address_is_accepted_and_scheme_added():
    assert cm.validate_public_url("example.com/jobs/1") == "https://example.com/jobs/1"


def test_unknown_host_gives_clear_message(monkeypatch):
    def boom(*a, **k):
        raise socket.gaierror("no such host")
    monkeypatch.setattr(cm, "_resolve", boom)
    with pytest.raises(ValueError, match="could not be found"):
        cm.validate_public_url("https://no-such-host.example")


# ───────────────────────── reading pages ─────────────────────────
def test_structured_job_data_is_used():
    info = cm.extract_job_info(LD_JSON_PAGE)
    assert info["source"].startswith("structured")
    assert info["title"] == "Data Engineer" and info["company"] == "Acme Corp"
    assert "Python" in info["text"] and "3+ years" in info["text"]
    assert "<p>" not in info["text"]                       # escaped HTML was turned into plain text
    assert "unrelated text" not in info["text"]


def test_plain_page_drops_menus_scripts_and_footers():
    info = cm.extract_job_info(PLAIN_PAGE)
    assert "SQL" in info["text"] and "Tableau" in info["text"]
    for noise in ("MENU-LINK", "HEADER-TEXT", "FOOTER-TEXT", "secret", "color:red"):
        assert noise not in info["text"]


def test_requirements_are_extracted():
    r = cm.extract_requirements("Candidates need 3+ years of experience. Bachelor's degree preferred. 2-4 years experience with SQL. Master's is a plus.")
    assert "3+ years" in r["years"] and "2-4 years" in r["years"]
    assert "Bachelor's degree" in r["education"] and "Master's degree" in r["education"]


# ───────────────────────── fetching rules (HTTP replaced) ─────────────────────────
def fake_http(pages):
    """pages: {url: (status, headers, body)}; robots.txt answers 404 unless given."""
    def _get(url):
        if url in pages:
            return pages[url]
        if url.endswith("/robots.txt"):
            return 404, {}, b""
        return 404, {}, b""
    return _get


def test_fetch_reads_a_normal_page(monkeypatch):
    monkeypatch.setattr(cm, "_http_get", fake_http({"https://example.com/job": (200, {"content-type": "text/html; charset=utf-8"}, PLAIN_PAGE.encode())}))
    job = cm.fetch_job_page("https://example.com/job")
    assert job["url"] == "https://example.com/job" and "Tableau" in job["text"]


@pytest.mark.parametrize("status", [401, 403, 429])
def test_blocked_sites_give_paste_hint(monkeypatch, status):
    monkeypatch.setattr(cm, "_http_get", fake_http({"https://example.com/job": (status, {}, b"")}))
    with pytest.raises(ValueError, match="paste"):
        cm.fetch_job_page("https://example.com/job")


def test_missing_page(monkeypatch):
    monkeypatch.setattr(cm, "_http_get", fake_http({}))
    with pytest.raises(ValueError, match="not found"):
        cm.fetch_job_page("https://example.com/job")


def test_non_html_is_rejected(monkeypatch):
    monkeypatch.setattr(cm, "_http_get", fake_http({"https://example.com/job.pdf": (200, {"content-type": "application/pdf"}, b"%PDF")}))
    with pytest.raises(ValueError, match="not a web page"):
        cm.fetch_job_page("https://example.com/job.pdf")


def test_robots_txt_is_respected(monkeypatch):
    pages = {"https://example.com/robots.txt": (200, {}, b"User-agent: *\nDisallow: /jobs/"),
             "https://example.com/jobs/1": (200, {"content-type": "text/html"}, PLAIN_PAGE.encode())}
    monkeypatch.setattr(cm, "_http_get", fake_http(pages))
    with pytest.raises(ValueError, match="robots.txt"):
        cm.fetch_job_page("https://example.com/jobs/1")


def test_redirect_to_private_network_is_blocked(monkeypatch):
    def dns(host, port, type=None, **kw):
        ip = "10.0.0.5" if host == "internal.example" else "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]
    monkeypatch.setattr(cm, "_resolve", dns)
    pages = {"https://example.com/go": (302, {"location": "http://internal.example/admin"}, b"")}
    monkeypatch.setattr(cm, "_http_get", fake_http(pages))
    with pytest.raises(ValueError, match="private or internal"):
        cm.fetch_job_page("https://example.com/go")


def test_redirect_loop_is_stopped(monkeypatch):
    pages = {"https://example.com/a": (302, {"location": "https://example.com/a"}, b"")}
    monkeypatch.setattr(cm, "_http_get", fake_http(pages))
    with pytest.raises(ValueError, match="redirects"):
        cm.fetch_job_page("https://example.com/a")


def test_page_with_almost_no_text(monkeypatch):
    monkeypatch.setattr(cm, "_http_get", fake_http({"https://example.com/js": (200, {"content-type": "text/html"}, b"<html><body><div id=app></div></body></html>")}))
    with pytest.raises(ValueError, match="JavaScript or a login"):
        cm.fetch_job_page("https://example.com/js")


# ───────────────────────── resume versus job ─────────────────────────
def test_fit_score_matched_and_missing():
    r = cm.analyze_company_fit(["Python", "SQL"], JOB_TEXT, DICT)
    assert r["sufficient_data"] is True
    assert set(r["matched"]) >= {"Python", "SQL"}
    assert "Docker" in r["missing"] and "AWS" in r["missing"]
    assert 0 <= r["score"] <= 100 and r["formula"]
    assert "3+ years" in r["years"] and "Bachelor's degree" in r["education"]
    assert r["tips"][-1].startswith("Only add skills you genuinely have")
    assert "guarantee" in r["disclaimer"].lower()


def test_missing_skills_are_ranked_by_how_often_the_job_mentions_them():
    text = ("Python Python Python SQL SQL Docker. Spark. Tableau. AWS AWS. " * 3) + "Plenty of other words to make the text longer. " * 5
    r = cm.analyze_company_fit([], text, DICT)
    assert r["missing"][0] == "Python" and r["mentions"]["Python"] > r["mentions"]["Tableau"]


def test_too_few_job_skills_gives_clear_message():
    r = cm.analyze_company_fit(["Python"], "We value teamwork and good communication. " * 10, DICT)
    assert r["sufficient_data"] is False and "too few" in r["message"]


def test_full_match_has_no_missing_skills():
    r = cm.analyze_company_fit(DICT["skills"], JOB_TEXT, DICT)
    assert r["missing"] == [] and r["score"] == 100
