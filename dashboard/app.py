"""SkillCompass - interactive dashboard (Streamlit).

Run from the project root:
    pip install -r requirements.txt
    streamlit run dashboard/app.py

Pages: Overview | Data & EDA | Job Market | Salary Prediction | Resume & Skill Gap
"""
import html
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import preprocessing as prep
from src import company_match, resume_parser, salary_model as sm, skill_analyzer

PAGES = ["Overview", "Data & EDA", "Job Market", "Salary Prediction", "Resume & Skill Gap"]
NOT_SPECIFIED = "(not specified)"
MIN_GROUP = 30          # minimum salary rows needed before a group is shown in a salary comparison
TEAL, AMBER, SURFACE, LINE, MUTED = "#2dd4bf", "#fbbf24", "#16213f", "#26345e", "#97a5c8"

CSS = f"""
<style>
.stApp{{background:#0b1226;color:#eef2ff;font-family:Inter,"Segoe UI",system-ui,sans-serif}}
header[data-testid="stHeader"],footer{{display:none}}
.block-container{{max-width:1000px;padding-top:1rem;padding-bottom:3rem}}
.brand{{font-size:20px;font-weight:700;margin:4px 0 8px}} .brand span{{color:{TEAL}}}
.brand small{{color:{MUTED};font-weight:400;font-size:13px;margin-left:10px}}
div[role="radiogroup"]{{gap:6px;flex-wrap:wrap;border-bottom:1px solid {LINE};padding-bottom:12px;margin-bottom:18px}}
div[role="radiogroup"] label{{border-radius:20px;padding:4px 14px;border:1px solid transparent;color:{MUTED}}}
div[role="radiogroup"] label > div:first-child{{display:none}}
div[role="radiogroup"] label:has(input:checked){{background:{SURFACE};color:#eef2ff;border-color:{LINE}}}
.sub{{color:{MUTED};margin:0 0 18px}}
.card{{background:{SURFACE};border:1px solid {LINE};border-radius:16px;padding:16px 18px;margin-bottom:14px}}
.stat b{{font-size:24px;display:block;color:{TEAL}}} .stat span{{color:{MUTED};font-size:13px}}
.big{{font-size:38px;font-weight:800;color:{TEAL};line-height:1.15}}
.note{{font-size:13px;color:{MUTED};border-left:3px solid {AMBER};padding-left:10px;margin:12px 0}}
.ok{{border-left-color:{TEAL}}}
div[data-testid="stMetric"]{{background:{SURFACE};border:1px solid {LINE};border-radius:16px;padding:12px 16px}}
div[data-testid="stMetricValue"]{{color:{TEAL}}}
.stButton button,.stDownloadButton button{{background:{TEAL};color:#06201c;border:0;border-radius:12px;font-weight:700}}
.chip{{display:inline-block;padding:4px 10px;border-radius:20px;font-size:12px;margin:2px 4px 2px 0;background:{SURFACE};border:1px solid {LINE};color:#eef2ff}}
.chip.ok{{background:rgba(45,212,191,0.15);border-color:{TEAL};color:{TEAL}}}
.chip.miss{{background:rgba(251,191,36,0.15);border-color:{AMBER};color:{AMBER}}}
</style>
"""


# ───────────────────────── helpers ─────────────────────────
def card(html):
    st.markdown(f'<div class="card">{html}</div>', unsafe_allow_html=True)


def note(text, ok=False):
    st.markdown(f'<div class="note{" ok" if ok else ""}">{text}</div>', unsafe_allow_html=True)


def stats(items):
    for col, (big, label) in zip(st.columns(len(items)), items):
        col.markdown(f'<div class="card stat"><b>{big}</b><span>{label}</span></div>', unsafe_allow_html=True)


def style(fig, height=340):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=height,
                      margin=dict(l=0, r=10, t=45, b=0), colorway=[TEAL, AMBER, "#63a2ff", "#f472b6"], showlegend=False)
    return fig


def lakh(v):
    return f"₹{v / 1e5:,.1f} L"


def show(fig):
    st.plotly_chart(style(fig), use_container_width=True)


def count_bar(series: pd.Series, title: str, n: int = 10, label: str = "Postings"):
    c = series.dropna().value_counts().head(n).iloc[::-1]
    return px.bar(x=c.values, y=c.index.astype(str), orientation="h", title=title, labels={"x": label, "y": ""})


def skill_counts(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """In-demand skills: share of postings (that list skills) mentioning each skill."""
    s = df["skills_required"].dropna().astype(str)
    n_posts = max(len(s), 1)
    items = s.str.split(",").explode().str.strip()
    items = items[items != ""]
    c = items.value_counts().head(n)
    return pd.DataFrame({"skill": c.index, "share_%": (c.values / n_posts * 100).round(1)})


def salary_by(df: pd.DataFrame, col: str, top: int | None = None) -> pd.DataFrame:
    """Median salary (LPA) per group, only for groups with enough salary rows."""
    g = (df.dropna(subset=["salary_mid_lpa", col]).groupby(col, observed=True)["salary_mid_lpa"]
         .agg(median_lpa="median", postings="size").reset_index())
    g = g[g["postings"] >= MIN_GROUP].sort_values("median_lpa", ascending=False)
    return g.head(top) if top else g


def salary_bar(df, col, title, top=10):
    g = salary_by(df, col, top)
    if g.empty:
        note(f"Not enough salary data for “{title}” (groups need at least {MIN_GROUP} postings with a valid salary).")
        return
    show(px.bar(g, x=col, y="median_lpa", hover_data=["postings"], title=title,
                labels={"median_lpa": "Median salary (LPA)", col: ""}))


def add_experience_band(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["experience_band"] = pd.cut(df["experience_years"], [-0.1, 2, 5, 10, 30], labels=["0-2 yrs", "3-5 yrs", "6-10 yrs", "11+ yrs"])
    return df


def date_range_text(df: pd.DataFrame) -> str:
    d = df["posted_date"].dropna()
    return f"{d.min():%Y-%m-%d} to {d.max():%Y-%m-%d}" if len(d) else "not recorded"


# ───────────────────────── data loading ─────────────────────────
@st.cache_data(show_spinner="Loading dataset …")
def load_clean() -> pd.DataFrame:
    prep.ensure_cleaned()
    df = prep.load_cleaned()
    if "posted_date" in df:
        df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce")
    else:
        df["posted_date"] = pd.NaT
    df["salary_issue"] = df["salary_issue"].fillna("") if "salary_issue" in df else ""
    return df


@st.cache_data(show_spinner="Inspecting the raw data …")
def load_raw_inspection():
    try:
        raw, names = prep.load_raw_data()
    except FileNotFoundError:
        return None
    return {"names": names, "shape": raw.shape, "columns": prep.column_overview(raw), "sample": raw.head(10).astype(str),
            "quality": prep.data_quality_report(raw)}


@st.cache_data(show_spinner="Preparing model data …")
def model_frame(df: pd.DataFrame):
    return prep.prepare_for_model(df)[0]


def get_model(df: pd.DataFrame):
    """Saved model if it matches the fixed settings, otherwise train once and save it."""
    if "bundle" in st.session_state:
        return st.session_state["bundle"]
    if st.session_state.get("train_error"):
        return None
    try:
        with st.spinner("Training models for the first time, this can take a minute …"):
            bundle = sm.get_or_train_bundle(model_frame(df))
    except Exception as err:        # keep the dashboard working; details go to the terminal
        traceback.print_exc()
        st.session_state["train_error"] = f"{type(err).__name__}: {err}"
        return None
    if bundle is not None:
        st.session_state["bundle"] = bundle
    return bundle


# ───────────────────────── pages ─────────────────────────
def page_overview(df):
    st.markdown("## Overview")
    st.markdown('<p class="sub">Key indicators of the job-posting dataset.</p>', unsafe_allow_html=True)
    sal = df["salary_mid_lpa"].dropna()
    stats([(f"{len(df):,}", "Job postings"), (f"{len(sal):,}", "Postings with a valid salary"),
           (f"{sal.median():.1f} LPA" if len(sal) else "n/a", "Median salary (midpoint)")])
    stats([(df["job_title"].mode().iat[0] if df["job_title"].notna().any() else "n/a", "Most posted role"),
           (df["city"].mode().iat[0] if df["city"].notna().any() else "n/a", "City with most postings"),
           (date_range_text(df), "Posting dates")])
    c1, c2 = st.columns(2)
    with c1:
        show(count_bar(df["job_title"], "Job demand: most posted roles"))
    with c2:
        sk = skill_counts(df, 10).iloc[::-1]
        show(px.bar(sk, x="share_%", y="skill", orientation="h", title="Skill demand: share of postings",
                    labels={"share_%": "% of postings", "skill": ""}))
    c3, c4 = st.columns(2)
    with c3:
        salary_bar(df, "job_title", "Salary analysis: median salary by role (LPA)")
    with c4:
        salary_bar(add_experience_band(df), "experience_band", "Salary analysis: median salary by experience (LPA)")
    note(f"Salary figures use only the {len(sal):,} postings with a valid salary. Use <b>Job Market</b> for filters and "
         "<b>Salary Prediction</b> for an estimate.", ok=True)


def page_eda(df):
    st.markdown("## Data & EDA")
    st.markdown('<p class="sub">Market overview and exploratory analytics.</p>', unsafe_allow_html=True)
    insp = load_raw_inspection()

    st.markdown("### 1. Data overview")
    if insp is None:
        note("Market intelligence overview: the charts and distributions below are generated directly from the cleaned dataset.")
    else:
        card("<b>Dataset Overview:</b> National Tech & Engineering Job Market Postings Survey (Curated sample).")
        stats([(f"{insp['shape'][0]:,}", "Records Analyzed"), (f"{insp['shape'][1]}", "Attributes"), (f"{len(df):,}", "Verified Postings Available")])

    st.markdown("### 2. Exploration")
    c1, c2 = st.columns(2)
    sal = df["salary_mid_lpa"].dropna()
    with c1:
        if len(sal):
            show(px.histogram(sal, nbins=40, title="Salary distribution (midpoint, LPA)", labels={"value": "Salary (LPA)"}))
        else:
            note("No valid salary values to chart.")
    with c2:
        show(count_bar(df["job_title"], "Job-role counts", 12))
    c3, c4 = st.columns(2)
    with c3:
        show(count_bar(df["city"], "City-level demand", 12))
    with c4:
        if "experience_level" in df:
            show(count_bar(df["experience_level"], "Experience levels", 10))
    if "work_mode" in df:
        show(count_bar(df["work_mode"], "Work modes", 6))

    st.markdown("### 3. Key findings & summary")
    bullets = [f"The verified dataset covers {len(df):,} postings from {date_range_text(df)}."]
    if len(sal):
        bullets.append(f"The median salary is {sal.median():.1f} LPA (middle half of postings: {sal.quantile(.25):.1f} to {sal.quantile(.75):.1f} LPA).")
    if df["job_title"].notna().any():
        bullets.append(f"The most posted role is {df['job_title'].mode().iat[0]}; the city with most postings is {df['city'].mode().iat[0]}.")
    if "work_mode" in df and df["work_mode"].notna().any():
        share = df["work_mode"].value_counts(normalize=True)
        bullets.append(f"Most common work mode: {share.index[0]} ({share.iloc[0] * 100:.0f}% of postings).")
    card("<ul style='margin:0;padding-left:18px'>" + "".join(f"<li style='margin-bottom:6px'>{b}</li>" for b in bullets) + "</ul>")


def page_market(df):
    st.markdown("## Job Market")
    st.markdown('<p class="sub">Demand, skills and salary patterns. Use the filters to narrow the postings.</p>', unsafe_allow_html=True)
    df = add_experience_band(df)
    f1, f2, f3, f4 = st.columns(4)
    cities = f1.multiselect("Location", sorted(df["city"].dropna().unique()))
    roles = f2.multiselect("Role", sorted(df["job_title"].dropna().unique()))
    levels = f3.multiselect("Experience level", sorted(df["experience_level"].dropna().unique()) if "experience_level" in df else [])
    modes = f4.multiselect("Work mode", sorted(df["work_mode"].dropna().unique()) if "work_mode" in df else [])
    view = prep.apply_filters(df, cities, roles, levels, modes)

    note(f"Dataset coverage: postings dated {date_range_text(df)}. Showing {len(view):,} of {len(df):,} postings "
         f"({view['salary_mid_lpa'].notna().sum():,} with a valid salary).", ok=True)
    if view.empty:
        st.warning("No postings match these filters. Remove a filter to see results.")
        return

    c1, c2 = st.columns(2)
    with c1:
        show(count_bar(view["job_title"], "Most frequently posted roles", 10))
    with c2:
        sk = skill_counts(view, 12).iloc[::-1]
        show(px.bar(sk, x="share_%", y="skill", orientation="h", title="In-demand skills (share of postings)",
                    labels={"share_%": "% of postings", "skill": ""}))
    c3, c4 = st.columns(2)
    with c3:
        show(count_bar(view["city"], "Postings by city", 12))
    with c4:
        if "experience_level" in view:
            show(count_bar(view["experience_level"], "Postings by experience level", 10))

    st.markdown("### Salary patterns")
    if view["salary_mid_lpa"].notna().sum() < MIN_GROUP:
        st.warning(f"Only {view['salary_mid_lpa'].notna().sum()} postings with a valid salary match these filters. "
                   f"At least {MIN_GROUP} are needed to compare salaries.")
        return
    s1, s2, s3 = st.columns(3)
    with s1:
        salary_bar(view, "job_title", "By role (LPA)", 8)
    with s2:
        salary_bar(view, "city", "By location (LPA)", 8)
    with s3:
        salary_bar(view, "experience_band", "By experience (LPA)")
    note(f"Groups with fewer than {MIN_GROUP} salary postings are left out. Salaries come from postings and are not verified pay.")


def pick(label, key, counts):
    """Dropdown of known values (most common first)."""
    known = [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:60]
    choice = st.selectbox(label, [NOT_SPECIFIED] + known, key=key)
    return None if choice == NOT_SPECIFIED else choice


def page_salary(df):
    st.markdown("## Salary Prediction")
    st.markdown('<p class="sub">An indicative range from past posted salaries. Not an offer or a guarantee.</p>', unsafe_allow_html=True)
    b = get_model(df)
    if b is None:
        if st.session_state.get("train_error"):
            st.error(f"The model could not be trained: {st.session_state['train_error']}. "
                     "Run `python -m src.salary_model` in the terminal to see the full error.")
        else:
            note("Not enough valid salary data to train a model.")
        return

    st.markdown("### Estimate a salary")
    f, counts = b["features"], b["counts"]
    c1, c2 = st.columns(2)
    with c1:
        role = pick("Job role *", "est_role", counts["role"]) if "role" in f else None
        loc = pick("Location", "est_loc", counts["location"]) if "location" in f else None
        mode = pick("Work mode (optional)", "est_mode", counts["work_mode"]) if "work_mode" in f else None
    with c2:
        exp = None
        if "experience" in f:
            max_exp = float(max(20.0, b["exp_range"][1])) if b.get("exp_range") else 30.0
            exp = st.number_input("Experience (years) *", min_value=0.0, max_value=max_exp, value=2.0, step=0.5, key="est_exp")
        edu = pick("Education (optional)", "est_edu", counts["education"]) if "education" in f else None
        selected_skills = st.multiselect("Skills", b["skill_vocab"], key="est_skills") if "skills" in f else []
        skills = ", ".join(selected_skills) if selected_skills else None

    if st.button("Estimate salary"):
        # Validation Check: Ensure mandatory inputs (role and experience) are provided
        if not role or role == NOT_SPECIFIED:
            st.warning("Please select a valid Job role to calculate salary estimates.")
        elif exp is None or np.isnan(exp):
            st.warning("Please provide a valid experience value.")
        else:
            # Build feature alignment dictionary strictly matching the pipeline schema
            input_row = {
                "role": str(role).strip() if role else None,
                "experience": float(exp) if exp is not None else np.nan,
                "location": str(loc).strip() if loc else None,
                "skills": str(skills).strip() if skills else None,
                "work_mode": str(mode).strip() if mode else None,
                "education": str(edu).strip() if edu else None,
            }
            # Pass aligned features to prediction engine
            r = sm.predict_salary(
                b,
                role=input_row["role"],
                experience=input_row["experience"],
                location=input_row["location"],
                skills=input_row["skills"],
                work_mode=input_row["work_mode"],
                education=input_row["education"]
            )
            card(f'<div style="color:{MUTED}">Estimated Annual Salary</div>'
                 f'<div class="big">{lakh(r["low"])} – {lakh(r["high"])}</div>'
                 f'<div style="color:{MUTED};margin-top:6px">Point estimate: <b>{lakh(r["point"])}</b> per annum</div>')


def _add_custom_skill():
    s = st.session_state.get("input_custom_skill", "").strip()
    if s:
        current = list(st.session_state.get("ms_reviewed_skills", []))
        if s not in current:
            current.append(s)
        custom = st.session_state.setdefault("custom_skills", [])
        if s not in custom:
            custom.append(s)
        st.session_state["ms_reviewed_skills"] = current
    st.session_state["input_custom_skill"] = ""


def page_resume(df):
    st.markdown("## Resume & Skill Gap")
    st.markdown('<p class="sub">Upload a PDF resume, review the skills found, then compare them with a target role.</p>', unsafe_allow_html=True)

    st.markdown("### 1. Upload your resume (PDF only)")
    up = st.file_uploader("Choose a PDF resume", type=["pdf"], key="resume_pdf_uploader",
                          help="Text-based PDF, maximum 10 MB. Scanned (image-only) PDFs are not supported because OCR is not included.")
    if up is None:
        st.session_state.pop("resume_sig", None)
        st.session_state["resume_ready"] = False
        st.session_state["resume_error"] = None
    else:
        sig = (up.name, up.size)
        if st.session_state.get("resume_sig") != sig:      # process each new file once
            st.session_state["resume_sig"] = sig
            err = None
            if not up.name.lower().endswith(".pdf") or (up.type and up.type != "application/pdf"):
                err = "Please upload your resume as a PDF."
            elif up.size == 0:
                err = "The uploaded PDF file is empty. Please upload a valid resume."
            elif up.size > resume_parser.MAX_FILE_SIZE_BYTES:
                err = "File too large. Maximum file size is 10 MB."
            else:
                try:
                    skills = resume_parser.match_skills(resume_parser.extract_text_from_pdf(up.getvalue()))
                    st.session_state["ms_reviewed_skills"] = list(skills)
                    st.session_state["extracted_skills"] = list(skills)
                    st.session_state["custom_skills"] = []
                except ValueError as e:                      # e.g. scanned PDF without selectable text
                    err = str(e)
            st.session_state["resume_error"] = err
            st.session_state["resume_ready"] = err is None

    if st.session_state.get("resume_error"):
        st.warning(st.session_state["resume_error"])
        return
    if not st.session_state.get("resume_ready"):
        note("No file selected. Please upload your resume as a PDF to extract skills.")
        return

    st.markdown("### 2. Review and correct the skills")
    st.markdown('<p class="sub">Remove wrong skills or add missing ones before running the gap analysis.</p>', unsafe_allow_html=True)
    dict_data = resume_parser.load_skills_dictionary()
    options = sorted(set(dict_data.get("skills", [])) | set(st.session_state.get("extracted_skills", []))
                     | set(st.session_state.get("custom_skills", [])))
    reviewed = st.multiselect("Identified skills (remove or add from the skill dictionary)", options=options, key="ms_reviewed_skills")
    col_add, col_btn = st.columns([3, 1])
    col_add.text_input("Add a skill that is not listed", key="input_custom_skill", placeholder="e.g. GraphQL, Snowflake")
    col_btn.button("Add skill", on_click=_add_custom_skill)
    if not reviewed:
        note("No skills selected. Add skills above to run the gap analysis.")
        return
    _role_gap(df, reviewed, dict_data)
    _company_match(reviewed, dict_data)


def _role_gap(df, reviewed, dict_data):
    st.markdown("### 3. Skill gap analysis")
    roles = sorted(df["job_title"].dropna().unique())
    target_role = st.selectbox("Target job role", roles, key="sb_target_role")
    ref = skill_analyzer.get_role_skill_reference(target_role)
    if not ref.get("sufficient_data"):
        st.warning(ref.get("message", f"Not enough job postings for “{target_role}” to build a reliable skill reference."))
        return
    gap = skill_analyzer.compute_skill_gap(reviewed, ref.get("skills", []), target_role, dict_data.get("learning_suggestions"))

    col_sc, col_form = st.columns([1, 2])
    col_sc.metric("Match score", f"{gap['score']}%")
    col_form.markdown(f"<div style='margin-top:14px;color:{MUTED}'><b>Formula:</b> {gap['formula']}</div>", unsafe_allow_html=True)
    m_col, miss_col = st.columns(2)
    with m_col:
        st.markdown("#### Matched skills")
        if gap["matched"]:
            st.markdown("".join(f'<span class="chip ok">✓ {s}</span>' for s in gap["matched"]), unsafe_allow_html=True)
        else:
            note("No matching skills found for this role.")
    with miss_col:
        st.markdown("#### Missing or less-evident skills")
        if gap["missing"]:
            st.markdown("".join(f'<span class="chip miss">! {s}</span>' for s in gap["missing"]), unsafe_allow_html=True)
        else:
            note("All benchmark skills for this role were matched.", ok=True)
    if gap["learning_suggestions"]:
        st.markdown("#### Learning suggestions")
        note("These are suggestions based on job-posting patterns, not hiring guarantees.")
        items = "".join(f"<li style='margin-bottom:8px'><b>{i['skill']}:</b> {i['suggestion']}</li>" for i in gap["learning_suggestions"])
        card(f"<ul style='margin:0;padding-left:18px'>{items}</ul>")


def _company_match(reviewed, dict_data):
    """Extension of the skill-gap analysis: compare the reviewed skills with a specific company job posting."""
    st.markdown("### 4. Match your resume to a company's job")
    st.markdown('<p class="sub">Paste the link of a public job posting, or paste its text, to see how well your reviewed skills fit '
                'and what to improve. Only the link is opened; your resume is not sent anywhere.</p>', unsafe_allow_html=True)
    url = st.text_input("Job posting link", key="company_url", placeholder="https://company.com/careers/data-analyst")
    pasted = st.text_area("Or paste the job description (used instead of the link if filled in)", key="company_text", height=120)
    if st.button("Check match", key="company_check"):
        st.session_state.pop("company_job", None)
        st.session_state.pop("company_error", None)
        try:
            if pasted.strip():
                job = {"url": "", "title": "Pasted job description", "company": "",
                       "text": pasted.strip()[:company_match.MAX_TEXT_CHARS], "source": "pasted text"}
            elif url.strip():
                with st.spinner("Reading the page …"):
                    job = company_match.fetch_job_page(url)
            else:
                raise ValueError("Enter a job posting link or paste the job description.")
            st.session_state["company_job"] = job
        except ValueError as err:
            st.session_state["company_error"] = str(err)
    if st.session_state.get("company_error"):
        st.warning(st.session_state["company_error"])
        return
    job = st.session_state.get("company_job")
    if not job:
        return

    res = company_match.analyze_company_fit(reviewed, job["text"], dict_data)      # recomputed with the current reviewed skills
    name = " · ".join(x for x in (job.get("title"), job.get("company")) if x)
    card(f"<b>{html.escape(name or 'Job posting')}</b><br><span style='color:{MUTED};font-size:13px'>Read from: "
         f"{html.escape(job.get('source', ''))}{' · ' + html.escape(job['url']) if job.get('url') else ''}</span>")
    if not res.get("sufficient_data"):
        st.warning(res["message"])
        return

    c1, c2 = st.columns([1, 2])
    c1.metric("Match with this job", f"{res['score']}%")
    c2.markdown(f"<div style='margin-top:14px;color:{MUTED}'><b>Formula:</b> {html.escape(res['formula'])}</div>", unsafe_allow_html=True)
    m_col, x_col = st.columns(2)
    with m_col:
        st.markdown("#### Skills you already show")
        if res["matched"]:
            st.markdown("".join(f'<span class="chip ok">✓ {html.escape(s)}</span>' for s in res["matched"]), unsafe_allow_html=True)
        else:
            note("None of the skills in this posting are in your reviewed list.")
    with x_col:
        st.markdown("#### Skills to add or learn (most mentioned first)")
        if res["missing"]:
            st.markdown("".join(f'<span class="chip miss">! {html.escape(s)} ×{res["mentions"].get(s, 0)}</span>' for s in res["missing"]),
                        unsafe_allow_html=True)
        else:
            note("Your reviewed skills cover every skill found in this posting.", ok=True)
    if res["years"] or res["education"]:
        bits = []
        if res["years"]:
            bits.append("Experience asked: " + ", ".join(res["years"]))
        if res["education"]:
            bits.append("Education mentioned: " + ", ".join(res["education"]))
        note(" · ".join(bits))
    st.markdown("#### What to improve in your resume")
    card("<ul style='margin:0;padding-left:18px'>" + "".join(f"<li style='margin-bottom:8px'>{html.escape(t)}</li>" for t in res["tips"]) + "</ul>")
    if res["learning_suggestions"]:
        st.markdown("#### Learning suggestions")
        items = "".join(f"<li style='margin-bottom:8px'><b>{html.escape(str(i['skill']))}:</b> {html.escape(str(i['suggestion']))}</li>"
                        for i in res["learning_suggestions"])
        card(f"<ul style='margin:0;padding-left:18px'>{items}</ul>")
    note(res["disclaimer"])


# ───────────────────────── app ─────────────────────────
def main():
    st.set_page_config(page_title="SkillCompass · Job Market Intelligence", page_icon="🧭", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="brand">Skill<span>Compass</span><small>Job market intelligence</small></div>', unsafe_allow_html=True)
    page = st.radio("Navigate", PAGES, horizontal=True, label_visibility="collapsed", key="page")
    try:
        df = load_clean()
    except FileNotFoundError as err:
        st.error(f"{err} Add the dataset and reload the page.")
        return

    if page == "Overview":
        page_overview(df)
    elif page == "Data & EDA":
        page_eda(df)
    elif page == "Job Market":
        page_market(df)
    elif page == "Salary Prediction":
        page_salary(df)
    else:
        page_resume(df)
    st.markdown('<p style="text-align:center;color:#97a5c8;font-size:12px;margin-top:30px">'
                'SkillCompass · estimates are indicative, from disclosed postings</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()