"""
app.py
======
Streamlit-powered Fake News Detection UI.

Run:
    streamlit run app.py
"""

import os
import pickle
import pathlib
import subprocess
import sys

import streamlit as st
import numpy as np

from preprocess import clean_text

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0a0c10;
    color: #e8eaf0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #10131a;
    border-right: 1px solid #1e2535;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Syne', sans-serif;
    color: #7eb8f7;
}

/* ── Header ── */
.main-header {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #7eb8f7 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.sub-header {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #6b7280;
    margin-bottom: 2rem;
    letter-spacing: 0.3px;
}

/* ── Cards ── */
.card {
    background: #13161f;
    border: 1px solid #1e2535;
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.2rem;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.8rem;
}

/* ── Result badges ── */
.result-real {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 50px;
    padding: 0.8rem 2rem;
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #34d399;
    margin: 1rem 0;
}
.result-fake {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    border-radius: 50px;
    padding: 0.8rem 2rem;
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #fca5a5;
    margin: 1rem 0;
}

/* ── Confidence bar ── */
.conf-bar-wrap {
    background: #1e2535;
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
    margin: 6px 0 2px;
}
.conf-bar-fill-real {
    height: 100%;
    background: linear-gradient(90deg, #10b981, #34d399);
    border-radius: 8px;
    transition: width 0.6s ease;
}
.conf-bar-fill-fake {
    height: 100%;
    background: linear-gradient(90deg, #ef4444, #f87171);
    border-radius: 8px;
    transition: width 0.6s ease;
}

/* ── Model accuracy pills ── */
.acc-pill {
    display: inline-block;
    background: #1e2535;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 0.5rem 1.1rem;
    font-size: 0.85rem;
    margin: 0.25rem;
    color: #a0aec0;
}
.acc-pill span {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    color: #7eb8f7;
}

/* ── Tips box ── */
.tip-box {
    background: #0f1520;
    border-left: 3px solid #7eb8f7;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    font-size: 0.88rem;
    color: #9ca3af;
    margin-top: 0.8rem;
}

/* ── Streamlit overrides ── */
.stTextArea textarea {
    background: #13161f !important;
    border: 1px solid #2d3748 !important;
    border-radius: 12px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: #7eb8f7 !important;
    box-shadow: 0 0 0 2px rgba(126, 184, 247, 0.15) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    padding: 0.6rem 2.5rem !important;
    font-size: 0.95rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stSelectbox > div > div {
    background: #13161f !important;
    border-color: #2d3748 !important;
    color: #e8eaf0 !important;
}
div[data-testid="metric-container"] {
    background: #13161f;
    border: 1px solid #1e2535;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
div[data-testid="metric-container"] label { color: #6b7280 !important; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    color: #7eb8f7 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_DIR = "models"

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load vectorizer + both models from disk. Returns None on missing files."""
    paths = {
        "vectorizer": os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"),
        "lr":         os.path.join(MODEL_DIR, "logistic_regression.pkl"),
        "nb":         os.path.join(MODEL_DIR, "naive_bayes.pkl"),
        "meta":       os.path.join(MODEL_DIR, "metadata.pkl"),
    }
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        return None, None, None, None

    with open(paths["vectorizer"], 'rb') as f: vec  = pickle.load(f)
    with open(paths["lr"],         'rb') as f: lr   = pickle.load(f)
    with open(paths["nb"],         'rb') as f: nb   = pickle.load(f)
    with open(paths["meta"],       'rb') as f: meta = pickle.load(f)
    return vec, lr, nb, meta


def predict(text: str, model, vectorizer):
    """Clean → vectorize → predict. Returns (label_str, confidence_pct, prob_array)."""
    cleaned = clean_text(text)
    if not cleaned.strip():
        return None, None, None
    vec  = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "REAL" if pred == 1 else "FAKE"
    conf  = float(np.max(prob)) * 100
    return label, conf, prob


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Fake News\nDetector")
    st.markdown("---")

    st.markdown("### Model")
    vec, lr_model, nb_model, meta = load_artifacts()

    model_choice = st.selectbox(
        "Choose classifier",
        ["Logistic Regression", "Naive Bayes"],
        index=0
    )
    active_model = lr_model if model_choice == "Logistic Regression" else nb_model

    if meta:
        st.markdown("### Accuracy on Test Set")
        st.markdown(
            f'<div class="acc-pill">Logistic Regression <span>{meta["lr_accuracy"]*100:.1f}%</span></div>'
            f'<div class="acc-pill">Naive Bayes <span>{meta["nb_accuracy"]*100:.1f}%</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
<div style='font-size:0.82rem; color:#6b7280; line-height:1.7;'>
Built with<br>
• <b style='color:#a0aec0'>scikit-learn</b> ML models<br>
• <b style='color:#a0aec0'>NLTK</b> text preprocessing<br>
• <b style='color:#a0aec0'>TF-IDF</b> feature extraction<br>
• <b style='color:#a0aec0'>Streamlit</b> UI
</div>
""", unsafe_allow_html=True)


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">Fake News Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Paste any news article below and let AI decide: Real or Fake?</div>',
    unsafe_allow_html=True
)

# ── Training gate ─────────────────────────────────────────────────────────────
if vec is None:
    st.warning("⚠️  Models not found. Click the button below to train them first.")
    if st.button("🚀 Train Models Now"):
        with st.spinner("Training models… (first run may take ~1 min)"):
            result = subprocess.run(
                [sys.executable, "train.py"],
                capture_output=True, text=True
            )
        if result.returncode == 0:
            st.success("✅ Training complete! Reload the page.")
            st.code(result.stdout[-2000:])
        else:
            st.error("Training failed.")
            st.code(result.stderr[-2000:])
    st.stop()

# ── Input area ────────────────────────────────────────────────────────────────
col_input, col_result = st.columns([3, 2], gap="large")

with col_input:
    st.markdown('<div class="card-title">📰 News Article Input</div>', unsafe_allow_html=True)
    news_text = st.text_area(
        label="",
        placeholder="Paste the news headline or full article text here…",
        height=280,
        key="news_input",
        label_visibility="collapsed"
    )
    word_count = len(news_text.split()) if news_text.strip() else 0
    st.caption(f"Word count: {word_count}")

    analyse_btn = st.button("🔍 Analyse Article", use_container_width=True)

    # Example news
    with st.expander("📋 Load example articles"):
        ex_col1, ex_col2 = st.columns(2)
        with ex_col1:
            if st.button("📰 Real news example"):
                st.session_state["news_input"] = (
                    "The Federal Reserve raised its benchmark interest rate by a quarter "
                    "percentage point on Wednesday, the latest step in its effort to bring "
                    "inflation down from a 40-year high without pushing the economy into recession. "
                    "The move brings the fed funds rate to a range of 5.25% to 5.5%, the highest "
                    "level in 22 years."
                )
                st.rerun()
        with ex_col2:
            if st.button("🚨 Fake news example"):
                st.session_state["news_input"] = (
                    "SHOCKING: Government scientists finally admit that chemtrails are real and "
                    "have been deliberately spraying the population with mind-control chemicals "
                    "for decades. Whistleblower insider reveals the deep state globalist agenda "
                    "to reduce world population by 90% using 5G towers and vaccine microchips. "
                    "Share before this gets taken down!!"
                )
                st.rerun()

    st.markdown("""
<div class="tip-box">
💡 <b>Tips for best results</b><br>
• Paste at least 2–3 sentences for higher confidence.<br>
• Both headline-only and full-article text work.<br>
• The model is trained on English-language news.
</div>
""", unsafe_allow_html=True)


with col_result:
    st.markdown('<div class="card-title">🎯 Detection Result</div>', unsafe_allow_html=True)

    if analyse_btn and news_text.strip():
        with st.spinner("Analysing…"):
            label, conf, probs = predict(news_text, active_model, vec)

        if label is None:
            st.error("Could not process the text. Please try a longer article.")
        else:
            # ── Verdict badge ──────────────────────────────────────────────
            if label == "REAL":
                st.markdown(
                    '<div class="result-real">✅ REAL NEWS</div>',
                    unsafe_allow_html=True
                )
                fill_class = "conf-bar-fill-real"
            else:
                st.markdown(
                    '<div class="result-fake">🚨 FAKE NEWS</div>',
                    unsafe_allow_html=True
                )
                fill_class = "conf-bar-fill-fake"

            # ── Confidence bar ─────────────────────────────────────────────
            st.markdown(f"**Confidence: {conf:.1f}%**")
            st.markdown(f"""
<div class="conf-bar-wrap">
  <div class="{fill_class}" style="width:{conf:.1f}%"></div>
</div>
<span style="font-size:0.75rem; color:#6b7280;">{conf:.1f}% confident</span>
""", unsafe_allow_html=True)

            st.markdown("---")

            # ── Probability breakdown ──────────────────────────────────────
            st.markdown("**Probability Breakdown**")
            m1, m2 = st.columns(2)
            with m1:
                st.metric("🚨 Fake", f"{probs[0]*100:.1f}%")
            with m2:
                st.metric("✅ Real", f"{probs[1]*100:.1f}%")

            # ── Confidence interpretation ──────────────────────────────────
            st.markdown("---")
            st.markdown("**Confidence Level**")
            if conf >= 90:
                st.success("🟢 Very High – strong signal detected")
            elif conf >= 75:
                st.info("🔵 High – model is fairly certain")
            elif conf >= 60:
                st.warning("🟡 Moderate – treat with caution")
            else:
                st.error("🔴 Low – borderline case, verify manually")

            # ── Model used ─────────────────────────────────────────────────
            st.caption(f"Model used: {model_choice}")

    elif analyse_btn:
        st.warning("Please enter some news text first.")
    else:
        st.markdown("""
<div style='text-align:center; padding: 3rem 1rem; color: #374151;'>
  <div style='font-size:3.5rem; margin-bottom:1rem;'>🔎</div>
  <div style='font-family:Syne,sans-serif; font-size:1rem; font-weight:600; color:#4b5563;'>
    Awaiting Input
  </div>
  <div style='font-size:0.83rem; margin-top:0.5rem; color:#374151;'>
    Enter a news article and click<br>"Analyse Article"
  </div>
</div>
""", unsafe_allow_html=True)


# ── How it works section ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="card-title">⚙️ How It Works</div>', unsafe_allow_html=True)

h1, h2, h3, h4 = st.columns(4)
steps = [
    ("1️⃣", "Text Preprocessing", "Lowercase, remove URLs, punctuation & stopwords, then lemmatize"),
    ("2️⃣", "TF-IDF Vectorization", "Convert cleaned text into a sparse numerical feature matrix"),
    ("3️⃣", "ML Classification", "Logistic Regression or Naive Bayes outputs a probability score"),
    ("4️⃣", "Verdict", "Threshold at 50% — above is Real, below is Fake"),
]
for col, (icon, title, desc) in zip([h1, h2, h3, h4], steps):
    with col:
        st.markdown(f"""
<div class="card" style="text-align:center; min-height:130px;">
  <div style="font-size:2rem;">{icon}</div>
  <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:0.9rem;
              margin: 0.4rem 0; color:#a0aec0;">{title}</div>
  <div style="font-size:0.78rem; color:#6b7280; line-height:1.5;">{desc}</div>
</div>
""", unsafe_allow_html=True)

# ── Confusion matrix image ─────────────────────────────────────────────────────
cm_path = os.path.join(MODEL_DIR, "confusion_matrices.png")
if os.path.exists(cm_path):
    st.markdown("---")
    st.markdown('<div class="card-title">📊 Model Evaluation — Confusion Matrices</div>',
                unsafe_allow_html=True)
    st.image(cm_path, use_container_width=True)

st.markdown("---")
st.caption("Fake News Detector · Built with scikit-learn, NLTK & Streamlit")