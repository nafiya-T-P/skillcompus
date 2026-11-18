"""AI-Powered Job Market Intelligence & Salary Prediction System (SkillCompass).

Streamlit application providing:
- Overview: Core KPI cards, market demand trends, and salary distributions.
- Data & EDA: Dataset inspection, cleaning logs, missing value reports, and figures.
- Job Market: Interactive filtering, regional and role demands, salary benchmarks (n >= 30).
- Salary Estimator: Indicative salary range and point estimate with model transparency.
- Resume & Skill Gap: PDF resume parser, skill extraction, target role skill gap analysis.
"""
from __future__ import annotations

import hashlib
import html
import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import streamlit as st

import resume_parser as rp
import salary_model as sm

# Constants & Paths
CLEANED_DATA_PATH = Path("data/processed/cleaned_jobs.csv")
CLEANING_LOG_PATH = Path("data/processed/cleaning_log.md")
EDA_REPORT_PATH = Path("reports/eda_report.md")
FIGURES_DIR = Path("reports/figures")

PAGES = ["Overview", "Data & EDA", "Job Market", "Salary Estimator", "Resume & Skill Gap"]

# Palette
TEAL = "#2dd4bf"
AMBER = "#fbbf24"
DARK_BG = "#0b1226"
CARD_BG = "#16213f"
BORDER = "#26345e"
MUTED = "#97a5c8"
TEXT = "#eef2ff"


# ───────────────────────── Helper Functions ─────────────────────────
def md(body: str) -> None:
    """Render raw HTML/markdown securely within Streamlit."""
    st.markdown(body, unsafe_allow_html=True)


def esc(text: object) -> str:
    """Escape text for safe HTML rendering."""
    return html.escape(str(text))


def lakh(val: float | None) -> str:
    """Format numeric INR value to Lakh INR string."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    return f"₹{val / 100_000:,.1f} L"


def chips(items: list[str], kind: str = "") -> str:
    """Render list of strings as styled badge chips."""
    kind_class = f" {kind}" if kind else ""
    return "".join(f'<span class="chip{kind_class}">{esc(s)}</span>' for s in items)


def style_fig(fig: go.Figure, height: int = 340) -> go.Figure:
    """Apply consistent styling across all Plotly charts."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(color=TEXT, family="Inter, system-ui, sans-serif"),
        colorway=[TEAL, AMBER, "#60a5fa", "#f472b6", "#a78bfa", "#34d399"],
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1e293b")
    fig.update_yaxes(showgrid=True, gridcolor="#1e293b")
    return fig


# ───────────────────────── Cached Data & Model Loaders ─────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame | None:
    """Load the single cleaned dataset."""
    if not CLEANED_DATA_PATH.exists():
        return None
    try:
        df = pd.read_csv(CLEANED_DATA_PATH, low_memory=False)
        return df
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading salary model...")
def get_bundle() -> dict | None:
    """Safely load the trained salary model bundle. Never raises traceback."""
    try:
        return sm.load_bundle()
    except Exception:
        return None


def apply_filters(df: pd.DataFrame, role: str, city: str, level: str, mode: str) -> pd.DataFrame:
    """Apply unified multi-attribute filter to dataset."""
    filtered = df.copy()
    if role and role != "All":
        filtered = filtered[filtered["job_title"] == role]
    if city and city != "All":
        filtered = filtered[filtered["city"] == city]
    if level and level != "All":
        filtered = filtered[filtered["experience_level"] == level]
    if mode and mode != "All":
        filtered = filtered[filtered["work_mode"] == mode]
    return filtered


# ───────────────────────── CSS Injection ─────────────────────────
CUSTOM_CSS = f"""
<style>
.stApp {{
    background-color: {DARK_BG};
    color: {TEXT};
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}}
header[data-testid="stHeader"] {{
    background-color: transparent;
}}
.block-container {{
    max-width: 1080px;
    padding-top: 1.5rem;
    padding-bottom: 3.5rem;
}}
.brand-title {{
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 2px;
}}
.brand-title span {{
    color: {TEAL};
}}
.brand-sub {{
    color: {MUTED};
    font-size: 13px;
    margin-bottom: 18px;
}}
.kpi-card {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
}}
.kpi-title {{
    color: {MUTED};
    font-size: 12px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
.kpi-value {{
    color: {TEAL};
    font-size: 26px;
    font-weight: 700;
    margin-top: 2px;
}}
.chip {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 16px;
    font-size: 12px;
    margin: 2px 4px 2px 0;
    background-color: {CARD_BG};
    border: 1px solid {BORDER};
    color: {TEXT};
}}
.chip.ok {{
    background-color: rgba(45, 212, 191, 0.15);
    border-color: {TEAL};
    color: {TEAL};
}}
.chip.miss {{
    background-color: rgba(251, 191, 36, 0.15);
    border-color: {AMBER};
    color: {AMBER};
}}
.info-banner {{
    background-color: rgba(38, 52, 94, 0.4);
    border-left: 3px solid {TEAL};
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 13px;
    color: {MUTED};
    margin: 12px 0 16px 0;
}}
.info-banner b {{
    color: {TEXT};
}}
div[data-testid="stSidebar"] {{
    background-color: #080d1c;
    border-right: 1px solid {BORDER};
}}
</style>
"""


# ───────────────────────── Page 1: Overview ─────────────────────────
def page_overview(df: pd.DataFrame | None):
    md('<div class="brand-title">Skill<span>Compass</span></div>')
    md('<div class="brand-sub">Job Market Intelligence, Salary Prediction & Resume Skill Gap Analysis</div>')

    md('<div class="info-banner"><b>Dataset Notice:</b> This project operates on <code>data/raw/india_job_market_messy_31150_records.csv</code>, '
       'a synthetic practice dataset for benchmarking. All metrics and predictions are indicative illustrations.</div>')

    if df is None:
        st.warning("Cleaned dataset not found. Please run `python prepare_dataset.py` to prepare the data.")
        return

    valid_sal = df["salary_mid_lpa"].dropna()
    dates = pd.to_datetime(df["posted_date"], errors="coerce").dropna()
    date_str = f"{dates.min().strftime('%b %Y')} – {dates.max().strftime('%b %Y')}" if not dates.empty else "N/A"

    # KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        md(f'<div class="kpi-card"><div class="kpi-title">Total Postings</div><div class="kpi-value">{len(df):,}</div></div>')
    with c2:
        md(f'<div class="kpi-card"><div class="kpi-title">Disclosed Salary</div><div class="kpi-value">{len(valid_sal):,}</div></div>')
    with c3:
        med_sal = f"₹{valid_sal.median():.1f} L" if len(valid_sal) else "N/A"
        md(f'<div class="kpi-card"><div class="kpi-title">Median Salary</div><div class="kpi-value">{med_sal}</div></div>')
    with c4:
        top_role = df["job_title"].mode().iat[0] if not df["job_title"].empty else "N/A"
        md(f'<div class="kpi-card"><div class="kpi-title">Top Role</div><div class="kpi-value" style="font-size:16px;line-height:2.1">{esc(top_role)}</div></div>')
    with c5:
        top_city = df["city"].mode().iat[0] if not df["city"].empty else "N/A"
        md(f'<div class="kpi-card"><div class="kpi-title">Top City</div><div class="kpi-value" style="font-size:16px;line-height:2.1">{esc(top_city)}</div></div>')

    md(f'<p style="color:{MUTED};font-size:12px;margin-top:2px;margin-bottom:20px">Posting date range: <b>{esc(date_str)}</b></p>')

    # 4 Compact Charts
    r1_col1, r1_col2 = st.columns(2)
    with r1_col1:
        top_roles = df["job_title"].value_counts().head(8).reset_index()
        top_roles.columns = ["Role", "Postings"]
        fig_roles = px.bar(top_roles, x="Postings", y="Role", orientation="h", title="Top Roles by Demand")
        fig_roles.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_fig(fig_roles), use_container_width=True)

    with r1_col2:
        city_counts = df["city"].value_counts().head(8).reset_index()
        city_counts.columns = ["City", "Postings"]
        fig_cities = px.bar(city_counts, x="City", y="Postings", title="Postings by Major Hub")
        st.plotly_chart(style_fig(fig_cities), use_container_width=True)

    r2_col1, r2_col2 = st.columns(2)
    with r2_col1:
        sal_clean = df.dropna(subset=["salary_mid_lpa"])
        fig_box = px.box(sal_clean, x="experience_level", y="salary_mid_lpa",
                         title="Salary Distribution by Experience Tier (LPA)",
                         labels={"experience_level": "Experience Tier", "salary_mid_lpa": "Annual Salary (LPA)"})
        st.plotly_chart(style_fig(fig_box), use_container_width=True)

    with r2_col2:
        # Skills frequency
        skills_series = df["skills_required"].dropna().str.split(",").explode().str.strip()
        skills_series = skills_series[skills_series != ""]
        top_sk = skills_series.value_counts().head(8).reset_index()
        top_sk.columns = ["Skill", "Mentions"]
        fig_skills = px.bar(top_sk, x="Mentions", y="Skill", orientation="h", title="Most In-Demand Skills")
        fig_skills.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_fig(fig_skills), use_container_width=True)


# ───────────────────────── Page 2: Data & EDA ─────────────────────────
def page_data_eda(df: pd.DataFrame | None):
    st.subheader("Data & Exploratory Data Analysis")
    st.caption("Detailed view of dataset structure, cleaning decisions, and distributional metrics.")

    md('<div class="info-banner"><b>Dataset Source:</b> <code>data/raw/india_job_market_messy_31150_records.csv</code><br>'
       'Cleaned and validated via <code>prepare_dataset.py</code> and logged to <code>data/processed/cleaning_log.md</code>.</div>')

    if df is None:
        st.warning("Cleaned dataset not found. Run: `python prepare_dataset.py`")
        return

    tab1, tab2, tab3 = st.tabs(["Dataset Sample & Fields", "Cleaning Decisions", "EDA Visualizations"])

    with tab1:
        st.markdown(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

        st.markdown("#### Missing Values by Column")
        missing_df = pd.DataFrame({
            "Column": df.columns,
            "Missing Count": df.isna().sum().values,
            "Missing %": (df.isna().sum().values / len(df) * 100).round(2),
        }).sort_values(by="Missing Count", ascending=False)
        st.dataframe(missing_df, use_container_width=True, hide_index=True)

    with tab2:
        if CLEANING_LOG_PATH.exists():
            st.markdown(CLEANING_LOG_PATH.read_text(encoding="utf-8"))
        else:
            st.info("Cleaning log not found. Run `python prepare_dataset.py` to regenerate `data/processed/cleaning_log.md`.")

    with tab3:
        if EDA_REPORT_PATH.exists():
            with st.expander("Read Full EDA Markdown Report", expanded=False):
                st.markdown(EDA_REPORT_PATH.read_text(encoding="utf-8"))

        fig_files = [
            ("Salary Distribution", FIGURES_DIR / "salary_distribution.png"),
            ("Top Demanded Roles", FIGURES_DIR / "top_roles.png"),
            ("City Demand", FIGURES_DIR / "city_demand.png"),
            ("Experience Levels and Work Modes", FIGURES_DIR / "experience_and_workmode.png"),
            ("Median Salary by Role (n >= 30)", FIGURES_DIR / "salary_by_role.png"),
        ]

        found_any = False
        for title, p in fig_files:
            if p.exists():
                found_any = True
                st.markdown(f"**{title}**")
                st.image(str(p), use_container_width=True)

        if not found_any:
            st.info("EDA figures not yet generated. Run `python eda.py` to build the charts and report.")


# ───────────────────────── Page 3: Job Market Intelligence ─────────────────────────
def page_market(df: pd.DataFrame | None):
    st.subheader("Job Market Intelligence")
    st.caption("Interactive analysis of market demand, roles, regional concentration, and salary benchmarks.")

    if df is None:
        st.warning("Cleaned dataset not found. Please run `python prepare_dataset.py`.")
        return

    # Filter section
    with st.expander("Filter Postings", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        role_opts = ["All"] + sorted(df["job_title"].dropna().unique().tolist())
        city_opts = ["All"] + sorted(df["city"].dropna().unique().tolist())
        level_opts = ["All"] + sorted(df["experience_level"].dropna().unique().tolist())
        mode_opts = ["All"] + sorted(df["work_mode"].dropna().unique().tolist())

        sel_role = f1.selectbox("Role", role_opts, key="mkt_role")
        sel_city = f2.selectbox("City", city_opts, key="mkt_city")
        sel_level = f3.selectbox("Experience Level", level_opts, key="mkt_level")
        sel_mode = f4.selectbox("Work Mode", mode_opts, key="mkt_mode")

    filtered = apply_filters(df, sel_role, sel_city, sel_level, sel_mode)

    valid_sal = filtered["salary_mid_lpa"].dropna()
    dates = pd.to_datetime(filtered["posted_date"], errors="coerce").dropna()
    date_str = f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}" if not dates.empty else "N/A"

    st.markdown(
        f"**Showing {len(filtered):,} postings** ({len(valid_sal):,} with valid salary) · "
        f"Dates: {esc(date_str)}"
    )

    if filtered.empty:
        st.info("No postings match the selected filter combination.")
        return

    m1, m2 = st.columns(2)
    with m1:
        top_roles = filtered["job_title"].value_counts().head(10).reset_index()
        top_roles.columns = ["Role", "Postings"]
        fig_r = px.bar(top_roles, x="Postings", y="Role", orientation="h", title="Top Roles in Selection")
        fig_r.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_fig(fig_r), use_container_width=True)

    with m2:
        top_cities = filtered["city"].value_counts().head(10).reset_index()
        top_cities.columns = ["City", "Postings"]
        fig_c = px.bar(top_cities, x="Postings", y="City", orientation="h", title="Top Cities in Selection")
        fig_c.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_fig(fig_c), use_container_width=True)

    st.markdown("### Salary Benchmarks by Dimension (Minimum n ≥ 30 Postings)")
    st.caption("Aggregates are shown strictly where sample size n ≥ 30 to prevent noise from small cohorts.")

    dim = st.radio("Group salary benchmarks by:", ["Job Role", "City", "Experience Level"], horizontal=True)
    col_map = {"Job Role": "job_title", "City": "city", "Experience Level": "experience_level"}
    group_col = col_map[dim]

    sal_sub = filtered.dropna(subset=["salary_mid_lpa"])
    agg = sal_sub.groupby(group_col)["salary_mid_lpa"].agg(
        Postings="count",
        Median="median",
        P25=lambda s: s.quantile(0.25),
        P75=lambda s: s.quantile(0.75),
        Mean="mean"
    ).reset_index()

    agg_sufficient = agg[agg["Postings"] >= 30].sort_values(by="Median", ascending=False)
    agg_insufficient = agg[agg["Postings"] < 30]

    if not agg_sufficient.empty:
        disp_agg = agg_sufficient.copy()
        for c in ["Median", "P25", "P75", "Mean"]:
            disp_agg[c] = disp_agg[c].map(lambda v: f"₹{v:.1f} L")
        st.dataframe(disp_agg, use_container_width=True, hide_index=True)

        fig_bench = px.bar(
            agg_sufficient,
            x=group_col,
            y="Median",
            title=f"Median Salary by {dim} (LPA)",
            labels={"Median": "Median Salary (LPA)"}
        )
        st.plotly_chart(style_fig(fig_bench), use_container_width=True)
    else:
        st.info("Insufficient data: no group has 30 or more salary postings under current filters.")

    if not agg_insufficient.empty:
        with st.expander(f"Groups with Insufficient Data (n < 30)", expanded=False):
            st.write(", ".join(f"{row[group_col]} (n={int(row['Postings'])})" for _, row in agg_insufficient.iterrows()))


# ───────────────────────── Page 4: Salary Estimator ─────────────────────────
def page_salary_estimator(df: pd.DataFrame | None):
    st.subheader("Salary Estimator")
    st.caption("Estimate indicative annual salary ranges based on regression models trained on job postings.")

    bundle = get_bundle()
    if bundle is None:
        st.warning("Model not trained. Run: `python train.py`")
        return

    # Check scikit-learn / python version mismatch
    bundle_sklearn = bundle.get("sklearn_version", "unknown")
    if bundle_sklearn != sklearn.__version__:
        st.info(f"Note: Model was trained with scikit-learn {bundle_sklearn} (current environment is {sklearn.__version__}).")

    counts = bundle.get("counts", {})
    features = bundle.get("features", [])

    c1, c2 = st.columns(2)
    with c1:
        roles_list = sorted(counts.get("role", {}).keys(), key=lambda r: -counts.get("role", {}).get(r, 0))
        role = st.selectbox("Target Role", roles_list if roles_list else ["Data Scientist"])

        locs_list = sorted(counts.get("location", {}).keys(), key=lambda l: -counts.get("location", {}).get(l, 0))
        loc = st.selectbox("Location / City", locs_list if locs_list else ["Bengaluru"])

        modes_list = sorted(counts.get("work_mode", {}).keys(), key=lambda m: -counts.get("work_mode", {}).get(m, 0))
        mode = st.selectbox("Work Mode", modes_list if modes_list else ["Hybrid"])

    with c2:
        exp_range = bundle.get("exp_range", (0.0, 30.0))
        exp = st.slider("Experience (years)", 0.0, float(exp_range[1]), 3.5, 0.5)

        edus_list = sorted(counts.get("education", {}).keys(), key=lambda e: -counts.get("education", {}).get(e, 0))
        edu = st.selectbox("Education Level", edus_list if edus_list else ["Bachelor's"])

        vocab = bundle.get("skill_vocab", [])
        dict_data = rp.load_skills_dictionary()
        # Canonical skill options
        all_skills = sorted(set(dict_data.get("skills", []) + vocab))
        chosen_skills = st.multiselect("Key Skills", all_skills, default=["Python", "SQL"] if "Python" in all_skills else [])

    prediction = sm.predict_salary(
        bundle=bundle,
        role=role,
        experience=exp,
        location=loc,
        skills=", ".join(chosen_skills),
        work_mode=mode,
        education=edu,
    )

    pt = prediction["point"]
    lo = prediction["low"]
    hi = prediction["high"]

    md(f"""
    <div class="kpi-card" style="margin-top:16px">
        <div class="kpi-title">Indicative Salary Range (Per Annum)</div>
        <div style="font-size:32px;font-weight:800;color:{TEAL};margin:4px 0 8px 0">
            {lakh(lo)} – {lakh(hi)}
        </div>
        <div style="color:{MUTED};font-size:14px">
            Point estimate: <b>{lakh(pt)}</b> per annum · Selected Model: <b>{esc(prediction['best_name'])}</b>
        </div>
    </div>
    """)

    for w in prediction["warnings"]:
        st.warning(w)

    md("""
    <div class="info-banner">
        <b>Limitations & Interpretability:</b>
        <ul style="margin:4px 0 0 0;padding-left:18px">
            <li><b>Synthetic Data:</b> The model is trained on a synthetic practice dataset; figures illustrate relative trends rather than verified employee pay.</li>
            <li><b>Disclosed Pay Only:</b> Estimates reflect what companies disclose in public postings, which may differ from negotiated compensation.</li>
            <li><b>Empirical Band:</b> The range is an empirical band (80% validation coverage) from held-out residual distributions, not a formal confidence interval.</li>
            <li><b>No Guarantee:</b> Predictions do not constitute binding salary offers or negotiation guarantees.</li>
        </ul>
    </div>
    """)

    with st.expander("How this was built (Candidate Models & Comparison)", expanded=False):
        res_df = bundle.get("results")
        if res_df is not None:
            show_df = res_df.copy()
            for col in ["cv_mae", "cv_rmse", "test_mae", "test_rmse"]:
                show_df[col] = show_df[col].map(lakh)
            for col in ["cv_r2", "test_r2"]:
                show_df[col] = show_df[col].round(3)
            st.dataframe(show_df, use_container_width=True, hide_index=True)
        st.write(f"**Training Settings:** 5-fold Cross-Validation · Seed 42 · 80/20 train/test split · Top 40 multi-hot skills")
        st.write(f"**Trained At (UTC):** {bundle.get('trained_at')} on {bundle.get('n_train', 0):,} training postings")


# ───────────────────────── Page 5: Resume & Skill Gap ─────────────────────────
def page_resume_skill_gap(df: pd.DataFrame | None):
    st.subheader("Resume & Skill Gap Analysis")
    st.caption("Upload a PDF resume to extract skills and compare your profile against target role requirements.")

    md('<div class="info-banner"><b>Privacy & Engine:</b> Resumes are parsed in-memory using PyMuPDF (pypdf fallback) '
       'and are never saved to disk. Scanned/image-only PDFs are not supported as OCR is not included.</div>')

    c_up, c_samp = st.columns([3, 1])
    with c_up:
        uploaded_file = st.file_uploader("Upload Resume (PDF format, up to 10 MB)", type=["pdf"], key="res_uploader")
    with c_samp:
        st.write("&nbsp;")
        use_sample = st.button("Use Sample Resume")

    pdf_bytes = None
    if use_sample:
        pdf_bytes = rp.sample_pdf_bytes()
        st.session_state["cached_pdf"] = ("sample_resume.pdf", pdf_bytes)
    elif uploaded_file is not None:
        pdf_bytes = uploaded_file.getvalue()
        st.session_state["cached_pdf"] = (uploaded_file.name, pdf_bytes)
    elif "cached_pdf" in st.session_state:
        _, pdf_bytes = st.session_state["cached_pdf"]

    if pdf_bytes is not None:
        try:
            raw_text = rp.extract_text_from_pdf(pdf_bytes)
            detected = rp.match_skills(raw_text)
            st.success(f"Successfully extracted {len(raw_text):,} characters of selectable text.")
        except ValueError as e:
            st.error(str(e))
            return

        st.markdown("### 1. Extracted Skills Review")
        st.caption("Verify and edit detected skills before running gap analysis.")

        dict_data = rp.load_skills_dictionary()
        all_skills = dict_data.get("skills", [])

        # Unique state tracking for skills edit
        current_skills = st.multiselect(
            "Your Skills (add missing skills or remove false positives):",
            all_skills,
            default=[s for s in detected if s in all_skills],
            key="reviewed_skills_multiselect"
        )

        st.markdown("### 2. Target Role Skill Gap Analysis")
        if df is not None and not df.empty:
            role_options = sorted(df["job_title"].dropna().unique().tolist())
        else:
            role_options = ["Data Scientist", "Software Engineer", "Data Engineer", "Machine Learning Engineer"]

        target_role = st.selectbox("Select Target Role for Comparison", role_options)

        gap_result = rp.calculate_skill_gap(
            candidate_skills=current_skills,
            target_role=target_role,
            dict_data=dict_data
        )

        if not gap_result["sufficient_data"]:
            st.info(gap_result["message"])
            return

        # Score display
        score_pct = gap_result["score_pct"]
        score_text = gap_result["score_text"]

        md(f"""
        <div class="kpi-card" style="margin-top:12px">
            <div class="kpi-title">Role Match Score</div>
            <div style="font-size:30px;font-weight:800;color:{TEAL};margin-top:2px">{score_pct}%</div>
            <div style="color:{MUTED};font-size:13px">{esc(score_text)} · {esc(gap_result['message'])}</div>
        </div>
        """)

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("#### Matched Skills")
            if gap_result["matched"]:
                md(chips(gap_result["matched"], "ok"))
            else:
                st.write("No matching skills found.")

        with g2:
            st.markdown("#### Missing or In-Demand Skills")
            if gap_result["missing"]:
                md(chips(gap_result["missing"], "miss"))
            else:
                st.write("All top reference skills matched.")

        if gap_result["suggestions"]:
            st.markdown("#### Learning Suggestions")
            st.caption("Suggestions, not hiring guarantees. Continuous practical projects build genuine mastery.")
            for item in gap_result["suggestions"]:
                st.markdown(f"- **{esc(item['skill'])}:** {esc(item['tip'])}")
    else:
        st.info("Upload a PDF resume or click 'Use Sample Resume' to begin skill extraction.")


# ───────────────────────── Main Navigation & Entry Point ─────────────────────────
def main():
    st.set_page_config(page_title="SkillCompass - Job Market & Salary Intelligence", page_icon="🧭", layout="wide")
    md(CUSTOM_CSS)

    df = load_data()

    # Sidebar Navigation
    with st.sidebar:
        md('<div class="brand-title" style="margin-bottom:8px">Skill<span>Compass</span></div>')
        md('<div class="brand-sub" style="margin-bottom:20px">Market Intelligence &amp; Prediction</div>')
        page = st.radio("Navigation", PAGES, label_visibility="collapsed")
        st.markdown("---")
        if df is not None:
            st.caption(f"Cleaned Postings: **{len(df):,}**")
            valid_sal_cnt = df["salary_mid_lpa"].notna().sum()
            st.caption(f"Salaried Postings: **{valid_sal_cnt:,}**")
        st.caption("Version 2.0 · Windows / Streamlit")

    if page == "Overview":
        page_overview(df)
    elif page == "Data & EDA":
        page_data_eda(df)
    elif page == "Job Market":
        page_market(df)
    elif page == "Salary Estimator":
        page_salary_estimator(df)
    elif page == "Resume & Skill Gap":
        page_resume_skill_gap(df)


if __name__ == "__main__":
    main()
