import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.preprocess import clean_text, extract_skills_from_text, get_top_keywords
from utils.text_extractor import extract_text_from_upload, load_csv_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Resume Screening AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }
    section[data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    .hero {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 45%, #ecfeff 100%);
        border: 1px solid #e2e8f0;
        border-radius: 24px;
        padding: 28px 30px;
        margin-bottom: 18px;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.03em;
    }
    .hero p {
        margin: 8px 0 0 0;
        color: #475569;
        font-size: 1rem;
        line-height: 1.6;
    }
    .card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 18px 18px 16px 18px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
    }
    .metric {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
    }
    .metric .value {
        font-size: 2rem;
        font-weight: 800;
        color: #1d4ed8;
        line-height: 1.1;
    }
    .metric .label {
        margin-top: 6px;
        color: #64748b;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin: 18px 0 10px 0;
    }
    .note-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 16px;
        padding: 14px 16px;
        color: #1e3a8a;
        line-height: 1.6;
    }
    .result-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #dbeafe;
        border-radius: 22px;
        padding: 24px;
        text-align: center;
    }
    .result-label {
        color: #64748b;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 8px;
    }
    .result-role {
        color: #0f172a;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0;
    }
    .pill {
        display: inline-block;
        padding: 5px 12px;
        margin: 4px 6px 0 0;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1d4ed8;
        border-radius: 999px;
        font-size: 0.82rem;
    }
    .tiny {
        color: #64748b;
        font-size: 0.85rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


@st.cache_resource(show_spinner=False)
def load_bundle():
    model_path = _first_existing([
        ROOT / "model" / "resume_model.pkl",
        ROOT / "model" / "resume_model.joblib",
    ])
    meta_path = _first_existing([
        ROOT / "model" / "model_meta.json",
    ])

    bundle = None
    meta = {}
    if model_path is not None:
        import joblib

        bundle = joblib.load(model_path)
    if meta_path is not None:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read model metadata: %s", exc)
            meta = {}
    return bundle, meta


@st.cache_data(show_spinner=False)
def load_resume_dataframe() -> pd.DataFrame:
    csv_path = _first_existing([
        ROOT / "resume" / "Resume.csv",
        ROOT / "resume" / "resume.csv",
        ROOT / "Resume.csv",
        ROOT / "resume.csv",
    ])
    if csv_path is None:
        return pd.DataFrame(columns=["text", "category"])

    df = load_csv_dataset(csv_path)
    if df.empty:
        return df

    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["category"] = df["category"].fillna("Unknown").astype(str)
    df = df[df["text"].str.strip().str.len() > 20].reset_index(drop=True)
    df["clean_text"] = df["text"].map(clean_text)
    df["word_count"] = df["clean_text"].str.split().str.len().fillna(0).astype(int)
    df["char_count"] = df["text"].str.len().fillna(0).astype(int)
    df["snippet"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip().str[:220]
    return df


@st.cache_data(show_spinner=False)
def build_top_words(df: pd.DataFrame, category: str | None = None, top_n: int = 15):
    subset = df
    if category and category != "All categories":
        subset = df[df["category"] == category]
    if subset.empty:
        return []

    counter = Counter()
    for text in subset["clean_text"].head(250):
        counter.update([w for w in text.split() if len(w) > 2])
    return counter.most_common(top_n)


@st.cache_data(show_spinner=False)
def build_model_comparison(meta: dict) -> pd.DataFrame:
    results = meta.get("results", {}) or {}
    rows = []
    for model_name, m in results.items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": float(m.get("accuracy", 0)),
                "Precision": float(m.get("precision", 0)),
                "Recall": float(m.get("recall", 0)),
                "F1": float(m.get("f1", 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("F1", ascending=False) if rows else pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1"])


def predict_resume_text(text: str, bundle):
    clean = clean_text(text)
    pipeline = bundle["pipeline"]
    label_encoder = bundle["label_encoder"]
    classes = list(label_encoder.classes_)

    pred_idx = pipeline.predict([clean])[0]
    predicted_label = label_encoder.inverse_transform([pred_idx])[0]

    scores = None
    confidence = None
    score_label = "score"

    if hasattr(pipeline, "predict_proba"):
        arr = pipeline.predict_proba([clean])[0]
        scores = dict(zip(classes, arr))
        confidence = float(arr.max())
        score_label = "probability"
    elif hasattr(pipeline, "decision_function"):
        raw = np.asarray(pipeline.decision_function([clean]))
        raw = raw.ravel()
        raw = raw - raw.max()
        exp = np.exp(raw)
        probs = exp / exp.sum() if exp.sum() else np.ones_like(exp) / len(exp)
        scores = dict(zip(classes, probs))
        confidence = float(probs.max())
        score_label = "relative score"
    else:
        scores = {predicted_label: 1.0}
        confidence = 1.0

    return predicted_label, confidence, score_label, scores, clean


bundle, meta = load_bundle()
resume_df = load_resume_dataframe()
comparison_df = build_model_comparison(meta)

if bundle is None:
    st.error("Saved model not found. Run the existing training script first so the app can load `model/resume_model.pkl`.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📄 Resume Screening AI")
    st.caption("A simple Streamlit app built around the pre-trained model bundle.")

    page = st.radio(
        "Navigate",
        ["Home", "EDA", "Model Comparison", "Predict"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    model_name = meta.get("best_model", bundle.get("best_model_name", "Saved model"))
    classes = meta.get("classes", [])
    st.markdown(
        f"""
        <div class="card">
            <div class="tiny">Loaded model</div>
            <div style="font-size:1.1rem;font-weight:800;color:#0f172a;">{model_name}</div>
            <div class="tiny" style="margin-top:6px;">{len(classes)} resume categories</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Home
# ──────────────────────────────────────────────────────────────────────────────
if page == "Home":
    st.markdown(
        """
        <div class="hero">
            <h1>Resume Screening AI</h1>
            <p>A resume classifier app</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    metrics = [
        (len(resume_df), "Resumes loaded"),
        (resume_df["category"].nunique() if not resume_df.empty else 0, "Categories"),
        (meta.get("train_size", "—"), "Train samples"),
        (meta.get("test_size", "—"), "Test samples"),
    ]
    for col, (value, label) in zip(cols, metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric">
                    <div class="value">{value}</div>
                    <div class="label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">What this app does</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            <div class="card">
                <b>EDA</b><br>
                Explore the dataset with interactive charts and sample resume previews. Get a feel for the categories, text lengths, common words, and more.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="card">
                <b>Prediction</b><br>
                Upload a resume file or paste text to get a category prediction from the trained model. See confidence scores, top alternatives, and extracted skills.
               
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Quick note</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="note-box">
            The existing training script already creates the model bundle and metadata in <code>model/</code>.
            This app just loads those files and presents them in a clearer, fresher interface.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# EDA
# ──────────────────────────────────────────────────────────────────────────────
elif page == "EDA":
    st.markdown(
        """
        <div class="hero">
            <h1>Exploratory Data Analysis</h1>
            
        </div>
        """,
        unsafe_allow_html=True,
    )

    if resume_df.empty:
        st.warning("No CSV dataset was found in the `resume/` folder.")
        st.stop()

    top_category = resume_df["category"].value_counts().idxmax()
    top_count = int(resume_df["category"].value_counts().max())
    avg_words = float(resume_df["word_count"].mean())
    median_words = float(resume_df["word_count"].median())

    cols = st.columns(4)
    eda_metrics = [
        (len(resume_df), "Rows"),
        (resume_df["category"].nunique(), "Unique roles"),
        (f"{avg_words:.0f}", "Avg. words"),
        (f"{median_words:.0f}", "Median words"),
    ]
    for col, (value, label) in zip(cols, eda_metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric">
                    <div class="value">{value}</div>
                    <div class="label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">A few quick observations</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="note-box">
            <b>{top_category}</b> shows up the most in this dataset with <b>{top_count}</b> resumes.
            The average resume length is around <b>{avg_words:.0f}</b> words, which suggests these are fairly detailed resumes.
            I also noticed that some categories are a lot smaller than others, so the class balance is not perfectly even.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Category distribution</div>', unsafe_allow_html=True)
        cat_counts = resume_df["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig = px.bar(
            cat_counts,
            x="Count",
            y="Category",
            orientation="h",
            text="Count",
            title=None,
        )
        fig.update_layout(height=640, margin=dict(l=10, r=10, t=10, b=10), yaxis_title=None, xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Resume length by category</div>', unsafe_allow_html=True)
        fig = px.box(
            resume_df,
            x="category",
            y="word_count",
            points=False,
            title=None,
        )
        fig.update_layout(height=640, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None, yaxis_title="Words")
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Top words after cleaning</div>', unsafe_allow_html=True)
    top_category_choice = st.selectbox(
        "Choose a category",
        ["All categories"] + sorted(resume_df["category"].unique().tolist()),
        index=0,
    )
    top_words = build_top_words(resume_df, category=top_category_choice, top_n=15)
    if top_words:
        words_df = pd.DataFrame(top_words, columns=["Word", "Count"])
        fig = px.bar(words_df, x="Count", y="Word", orientation="h")
        fig.update_layout(height=500, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No words available for this selection.")

    st.markdown('<div class="section-title">Sample resumes</div>', unsafe_allow_html=True)
    sample_cols = st.columns([1, 1])
    with sample_cols[0]:
        sample_category = st.selectbox(
            "Pick a category to inspect",
            sorted(resume_df["category"].unique().tolist()),
            index=0,
        )
    subset = resume_df[resume_df["category"] == sample_category].reset_index(drop=True)
    sample_idx = 0
    with sample_cols[1]:
        if len(subset) > 1:
            sample_idx = st.slider("Pick a sample", 0, len(subset) - 1, 0)

    sample_row = subset.iloc[sample_idx]
    st.markdown(
        f"""
        <div class="card">
            <div class="tiny">Category</div>
            <div style="font-size:1.15rem;font-weight:800;color:#0f172a;">{sample_row['category']}</div>
            <div class="tiny" style="margin-top:6px;">{sample_row['word_count']} words · {sample_row['char_count']} characters</div>
            <hr style="border:none;border-top:1px solid #e2e8f0;margin:14px 0;">
            <div style="line-height:1.7;color:#334155;">{sample_row['snippet']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Model comparison
# ──────────────────────────────────────────────────────────────────────────────
elif page == "Model Comparison":
    st.markdown(
        """
        <div class="hero">
            <h1>Model Comparison</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if comparison_df.empty:
        st.warning("No model results were found in the saved metadata.")
        st.stop()

    best_row = comparison_df.iloc[0]
    left, right, extra = st.columns(3)
    with left:
        st.markdown(
            f"""
            <div class="metric">
                <div class="value">{meta.get('best_model', best_row['Model'])}</div>
                <div class="label">Saved model in use</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
            <div class="metric">
                <div class="value">{best_row['F1']:.3f}</div>
                <div class="label">Top F1 score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with extra:
        st.markdown(
            f"""
            <div class="metric">
                <div class="value">{best_row['Accuracy']:.3f}</div>
                <div class="label">Top accuracy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Comparison table</div>', unsafe_allow_html=True)
    display_df = comparison_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.3f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Metrics at a glance</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="F1", x=comparison_df["Model"], y=comparison_df["F1"]))
    fig.add_trace(go.Bar(name="Accuracy", x=comparison_df["Model"], y=comparison_df["Accuracy"]))
    fig.update_layout(barmode="group", height=520, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None, yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">What is actually deployed?</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div>
            The app uses the saved bundle from <code>model/resume_model.pkl</code>.
            In the saved metadata, <b>{meta.get('best_model', 'the top-scoring model')}</b> is the best scoringmodel that was kept for prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Predict
# ──────────────────────────────────────────────────────────────────────────────
elif page == "Predict":
    st.markdown(
        """
        <div class="hero">
            <h1>Predict</h1>
            <p>Upload a resume or paste text to get a category prediction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_in, col_out = st.columns([1.05, 0.95])
    with col_in:
        input_mode = st.radio("Input mode", ["Upload file", "Paste text"], horizontal=True)
        raw_text = ""

        if input_mode == "Upload file":
            uploaded = st.file_uploader(
                "Upload a resume file",
                type=["pdf", "docx", "txt", "csv"],
                help="PDF, DOCX, TXT, or CSV",
            )
            if uploaded is not None:
                with st.spinner("Extracting text..."):
                    raw_text = extract_text_from_upload(uploaded)
                if raw_text:
                    st.success(f"Extracted {len(raw_text)} characters.")
                else:
                    st.warning("Text extraction returned nothing for this file.")
        else:
            raw_text = st.text_area(
                "Paste resume text here",
                height=260,
                placeholder="Paste the resume content here...",
            )

        example_text = ""
        if st.button("Load a sample text"):
            if not resume_df.empty:
                example_text = resume_df.iloc[0]["text"]
                st.session_state["sample_text"] = example_text
        if "sample_text" in st.session_state and not raw_text:
            raw_text = st.session_state["sample_text"]

        run_prediction = st.button("Predict category", type="primary", use_container_width=True)

    if run_prediction:
        if not raw_text.strip():
            st.warning("Add a resume first.")
            st.stop()

        with st.spinner("Scoring resume..."):
            predicted_label, confidence, score_label, scores, cleaned = predict_resume_text(raw_text, bundle)
            skills = extract_skills_from_text(raw_text)
            top_keywords = get_top_keywords(raw_text, top_n=10)

        if not predicted_label:
            st.error("Prediction failed.")
            st.stop()

        with col_out:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted category</div>
                    <div class="result-role">{predicted_label}</div>
                    <div style="margin-top:10px;color:#475569;">
                        Confidence: <b>{confidence:.1%}</b> ({score_label})
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="section-title">Top alternatives</div>', unsafe_allow_html=True)
            scored = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            for label, value in scored:
                st.markdown(
                    f"""
                    <div style="margin-bottom:10px;">
                        <div style="display:flex;justify-content:space-between;gap:12px;">
                            <span style="font-weight:{700 if label == predicted_label else 500};color:#0f172a;">{label}</span>
                            <span style="color:#64748b;">{value:.1%}</span>
                        </div>
                        <div style="height:8px;background:#e2e8f0;border-radius:999px;overflow:hidden;margin-top:6px;">
                            <div style="width:{value*100:.1f}%;height:100%;background:{'#2563eb' if label == predicted_label else '#94a3b8'};"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if raw_text.strip():
        st.markdown("---")
        left, right = st.columns(2)

        with left:
            st.markdown('<div class="section-title">Detected skills (simple keyword match)</div>', unsafe_allow_html=True)

            if skills:
                st.markdown(
                    "".join([f'<span class="pill">{skill}</span>' for skill in skills]),
                    unsafe_allow_html=True
                )
            else:
                st.caption("No common skills detected using simple keyword matching.")

            st.markdown('<div class="section-title">Top keywords from resume</div>', unsafe_allow_html=True)

            if top_keywords:
                kw_df = pd.DataFrame(top_keywords, columns=["Word", "Count"])
                fig = px.bar(kw_df, x="Count", y="Word", orientation="h")
                fig.update_layout(
                    height=360,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title=None,
                    yaxis_title=None
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Not enough text to extract keywords.")

        with right:
            st.markdown('<div class="section-title">Cleaned text preview</div>', unsafe_allow_html=True)

            cleaned = clean_text(raw_text)

            st.markdown(
                f"""
                <div class="card" style="max-height:360px;overflow:auto;line-height:1.75;color:#334155;white-space:pre-wrap;">
                    {cleaned[:1800]}{'...' if len(cleaned) > 1800 else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )
