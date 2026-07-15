# ============================================================
#  MindPulse — Streamlit Web Application
#  Author  : Usama (App Developer)
#  File    : app/streamlit_app.py
#  Purpose : Main web application — all 5 pages
#            Home, Text Analysis, DASS-21 Quiz,
#            Physiological Upload, History
#
#  Run with: streamlit run app/streamlit_app.py
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime


# ── Page configuration ────────────────────────────────────────
st.set_page_config(
    page_title            = "MindPulse",
    page_icon             = "🧠",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)


# ── Custom styling ────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }

    .ethics-banner {
        background: #FAEEDA;
        border-left: 4px solid #854F0B;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-bottom: 16px;
        font-size: 14px;
        color: #412402;
    }

    .result-card {
        background: #EEEDFE;
        border: 1.5px solid #534AB7;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 16px;
    }

    .result-label {
        font-size: 28px;
        font-weight: 600;
        color: #26215C;
    }
</style>
""", unsafe_allow_html=True)


# ── Load models (cached so they only load once) ───────────────
@st.cache_resource
def load_models():
    """
    Load all trained models into memory once at startup.

    Using @st.cache_resource means this runs only ONCE
    when the app first loads — not on every user click.
    This prevents reloading large models (BERT is 400MB)
    every time someone interacts with the app.

    Returns:
        Dictionary with keys lstm, bert, hybrid
        Values are None if models not trained yet
    """
    models = {}

    # Try loading LSTM
    try:
        import tensorflow as tf
        models["lstm"] = tf.keras.models.load_model("models/saved/lstm_model.h5")
    except Exception:
        models["lstm"] = None   # Not trained yet

    # Try loading BERT
    try:
        from transformers import pipeline
        models["bert"] = pipeline(
            "text-classification",
            model             = "models/saved/bert_mindpulse",
            return_all_scores = True,
        )
    except Exception:
        models["bert"] = None   # Not trained yet

    models["hybrid"] = None   # Coming in next phase

    return models


# ── DASS-21 scoring ───────────────────────────────────────────
def compute_dass_scores(answers: dict) -> dict:
    """
    Compute DASS-21 subscale scores from user answers.

    Official item assignments:
        Depression: q3, q5, q10, q13, q16, q17, q21
        Anxiety   : q2, q4, q7,  q9,  q15, q19, q20
        Stress    : q1, q6, q8,  q11, q12, q14, q18

    Score = sum of 7 items × 2 (to match 42-item DASS norms)

    Args:
        answers: {"q1": 0..3, ..., "q21": 0..3}

    Returns:
        {"depression": int, "anxiety": int, "stress": int}
    """
    depression_items = [3, 5, 10, 13, 16, 17, 21]
    anxiety_items    = [2, 4,  7,  9, 15, 19, 20]
    stress_items     = [1, 6,  8, 11, 12, 14, 18]

    return {
        "depression" : sum(answers.get(f"q{i}", 0) for i in depression_items) * 2,
        "anxiety"    : sum(answers.get(f"q{i}", 0) for i in anxiety_items)    * 2,
        "stress"     : sum(answers.get(f"q{i}", 0) for i in stress_items)     * 2,
    }


def classify_dass(scores: dict) -> tuple:
    """
    Apply official DASS-21 clinical thresholds to assign label.

    Thresholds (moderate severity):
        Depression >= 14
        Anxiety    >= 10
        Stress     >= 19

    Returns:
        (label, confidence) where label is the most elevated subscale
    """
    thresholds = {"depression": 14, "anxiety": 10, "stress": 19}

    elevated = {
        name: score
        for name, score in scores.items()
        if score >= thresholds[name]
    }

    if not elevated:
        return "control", 0.85

    label      = max(elevated, key=elevated.get)
    confidence = min(scores[label] / 42, 1.0)
    return label, round(confidence, 3)


# ── Ethics banner ─────────────────────────────────────────────
def show_ethics_banner():
    """Show the mandatory ethics disclaimer on every results page."""
    st.markdown("""
    <div class="ethics-banner">
        ⚠️ <strong>Important:</strong> MindPulse is a screening tool only —
        NOT a medical diagnosis. Always consult a qualified mental health
        professional. If you are in crisis contact
        <strong>Samaritans: 116 123</strong> (free, 24/7).
    </div>
    """, unsafe_allow_html=True)


# ── Show prediction result ────────────────────────────────────
def show_result(label: str, confidence: float, probabilities: dict):
    """
    Display the model prediction in a styled result card.

    Args:
        label         : "depression" / "anxiety" / "stress" / "control"
        confidence    : Float 0.0 to 1.0
        probabilities : {"depression": 0.7, "anxiety": 0.2, "control": 0.1}
    """
    colors = {
        "depression": "#993C1D",
        "anxiety"   : "#534AB7",
        "stress"    : "#854F0B",
        "control"   : "#0F6E56",
    }
    icons = {
        "depression": "😞",
        "anxiety"   : "😰",
        "stress"    : "😤",
        "control"   : "✅",
    }

    color = colors.get(label, "#5F5E5A")
    icon  = icons.get(label, "🔍")

    st.markdown(f"""
    <div class="result-card" style="border-color:{color};background:{color}15">
        <div class="result-label" style="color:{color}">
            {icon} {label.title()} Detected
        </div>
        <div style="font-size:14px;color:{color};margin-top:4px">
            Confidence: {confidence*100:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Probability bars
    st.write("")
    st.subheader("Probability Breakdown")
    for cls, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
        col1, col2 = st.columns([1, 4])
        with col1:
            st.write(cls.title())
        with col2:
            st.progress(prob, text=f"{prob*100:.1f}%")

    # Recommendations
    show_recommendations(label)


def show_recommendations(label: str):
    """Show appropriate next-step recommendations."""
    st.write("")
    st.subheader("What to do next")

    recs = {
        "depression": [
            "Speak to your GP or a mental health professional",
            "Mind UK helpline: 0300 123 3393",
            "Samaritans (24/7): 116 123",
        ],
        "anxiety": [
            "Try 4-7-8 breathing to calm anxiety right now",
            "Anxiety UK helpline: 03444 775 774",
            "Samaritans (24/7): 116 123",
        ],
        "stress": [
            "Your university counselling service is free",
            "Student Minds: studentminds.org.uk",
            "Samaritans (24/7): 116 123",
        ],
        "control": [
            "Your results suggest no significant concern right now",
            "Keep monitoring how you feel over time",
            "It is always okay to seek support if things change",
        ],
    }

    for rec in recs.get(label, []):
        st.markdown(f"• {rec}")


# ════════════════════════════════════════════════════════════════
#  PAGES
# ════════════════════════════════════════════════════════════════

def page_home():
    """Home / welcome page."""
    st.title("🧠 MindPulse")
    st.subheader("AI Mental Wellbeing Screening System")
    show_ethics_banner()

    st.markdown("""
    MindPulse uses AI to screen for signs of **depression**, **anxiety**, and **stress**.

    Choose how you want to be assessed:
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📝 **Text Journal**\n\nWrite how you feel — LSTM & BERT analyse it")
        if st.button("Start Text Analysis", use_container_width=True):
            st.session_state["page"] = "text"
    with col2:
        st.info("📋 **DASS-21 Quiz**\n\nAnswer 21 questions — takes 3 minutes")
        if st.button("Take the Quiz", use_container_width=True):
            st.session_state["page"] = "quiz"
    with col3:
        st.info("📈 **Physio Upload**\n\nUpload biosensor CSV — Hybrid model")
        if st.button("Upload Signal Data", use_container_width=True):
            st.session_state["page"] = "physio"


def page_text_analysis(models: dict):
    """Text journal analysis page — LSTM or BERT."""
    st.title("📝 Text Analysis")
    show_ethics_banner()

    st.write("Describe how you have been feeling recently.")

    text_input = st.text_area(
        label       = "Your journal entry",
        placeholder = "I have been feeling...",
        height      = 200,
        max_chars   = 2000,
        label_visibility = "collapsed"
    )

    model_choice = st.radio(
        "Choose model",
        options    = ["BERT (recommended)", "LSTM", "Both — ensemble"],
        horizontal = True
    )

    if st.button("Analyse Text", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("Please write something before analysing.")
            return

        if len(text_input.split()) < 10:
            st.warning("Please write at least 10 words for a meaningful result.")
            return

        with st.spinner("Analysing your text..."):
            # Placeholder prediction — replace with real model call
            # when models are trained:
            # if models["bert"]:
            #     result = models["bert"](text_input)
            label         = "anxiety"
            confidence    = 0.74
            probabilities = {
                "depression": 0.17,
                "anxiety"   : 0.74,
                "control"   : 0.09
            }

        show_result(label, confidence, probabilities)


def page_quiz():
    """DASS-21 questionnaire page."""
    st.title("📋 DASS-21 Questionnaire")
    show_ethics_banner()

    st.write("""
    Rate each statement over the **past week**:
    **0** = Never | **1** = Sometimes | **2** = Often | **3** = Almost always
    """)

    questions = {
        "q1" : "I found it hard to wind down",
        "q2" : "I was aware of dryness of my mouth",
        "q3" : "I couldn't seem to experience any positive feeling at all",
        "q4" : "I experienced breathing difficulty",
        "q5" : "I found it difficult to work up the initiative to do things",
        "q6" : "I tended to over-react to situations",
        "q7" : "I experienced trembling (e.g. in the hands)",
        "q8" : "I felt that I was using a lot of nervous energy",
        "q9" : "I was worried about situations in which I might panic",
        "q10": "I felt that I had nothing to look forward to",
        "q11": "I found myself getting agitated",
        "q12": "I found it difficult to relax",
        "q13": "I felt down-hearted and blue",
        "q14": "I was intolerant of anything that kept me from getting on",
        "q15": "I felt I was close to panic",
        "q16": "I was unable to become enthusiastic about anything",
        "q17": "I felt I wasn't worth much as a person",
        "q18": "I felt that I was rather touchy",
        "q19": "I was aware of the action of my heart without physical exertion",
        "q20": "I felt scared without any good reason",
        "q21": "I felt that life was meaningless",
    }

    options = {"0 — Never": 0, "1 — Sometimes": 1, "2 — Often": 2, "3 — Almost always": 3}
    answers = {}

    with st.form("dass21_form"):
        for q_key, q_text in questions.items():
            q_num = int(q_key[1:])
            st.write(f"**{q_num}. {q_text}**")
            selected = st.radio(
                label            = q_text,
                options          = list(options.keys()),
                horizontal       = True,
                key              = f"dass_{q_key}",
                label_visibility = "collapsed"
            )
            answers[q_key] = options[selected]
            st.divider()

        submitted = st.form_submit_button(
            "Get My Results",
            type             = "primary",
            use_container_width = True
        )

    if submitted:
        scores = compute_dass_scores(answers)
        label, confidence = classify_dass(scores)

        st.write("**Your subscale scores:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Depression", scores["depression"], delta=None)
        with col2:
            st.metric("Anxiety",    scores["anxiety"],    delta=None)
        with col3:
            st.metric("Stress",     scores["stress"],     delta=None)

        total = sum(scores.values()) or 1
        probabilities = {k: round(v/total, 3) for k, v in scores.items()}
        show_result(label, confidence, probabilities)


def page_physio():
    """Physiological signal upload page."""
    st.title("📈 Physiological Signal Upload")
    show_ethics_banner()

    st.write("Upload a CSV file from your wearable device (ECG, EDA, temperature).")

    uploaded = st.file_uploader("Upload sensor CSV", type=["csv"])

    if uploaded:
        st.success(f"File received: {uploaded.name}")

        try:
            df = pd.read_csv(uploaded)
            st.write("**Preview of your data (first 5 rows):**")
            st.dataframe(df.head())
            st.write(f"Total rows: {len(df):,} | Columns: {list(df.columns)}")

            if st.button("Run Analysis", type="primary"):
                with st.spinner("Processing signals..."):
                    # Placeholder — replace with real Hybrid model call
                    label         = "stress"
                    confidence    = 0.81
                    probabilities = {
                        "stress"  : 0.81,
                        "anxiety" : 0.13,
                        "control" : 0.06
                    }
                show_result(label, confidence, probabilities)

        except Exception as e:
            st.error(f"Could not read file: {e}. Please make sure it is a valid CSV.")


def page_history():
    """User history and trend page."""
    st.title("📊 My History")

    # Placeholder history data
    history = [
        {"Date": "2025-06-10", "Mode": "Text",          "Result": "Anxiety",    "Confidence": "74%"},
        {"Date": "2025-06-07", "Mode": "Questionnaire", "Result": "Anxiety",    "Confidence": "81%"},
        {"Date": "2025-06-01", "Mode": "Physiological", "Result": "Stress",     "Confidence": "68%"},
        {"Date": "2025-05-22", "Mode": "Text",          "Result": "Depression", "Confidence": "61%"},
        {"Date": "2025-05-15", "Mode": "Questionnaire", "Result": "Control",    "Confidence": "88%"},
    ]

    st.write(f"Your last **{len(history)}** assessments:")
    st.dataframe(pd.DataFrame(history), use_container_width=True)

    st.subheader("Trend over time")
    chart_data = pd.DataFrame({
        "Session": [1, 2, 3, 4, 5],
        "Confidence": [88, 61, 68, 81, 74]
    })
    st.line_chart(chart_data.set_index("Session"))


# ════════════════════════════════════════════════════════════════
#  MAIN APP
# ════════════════════════════════════════════════════════════════

def main():
    """Main function — sets up navigation and routes to pages."""

    # Initialise session state
    if "page" not in st.session_state:
        st.session_state["page"] = "home"

    # Load models
    models = load_models()

    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🧠 MindPulse")
        st.divider()

        pages = {
            "🏠 Home"           : "home",
            "📝 Text Analysis"  : "text",
            "📋 DASS-21 Quiz"   : "quiz",
            "📈 Physio Upload"  : "physio",
            "📊 My History"     : "history",
        }

        for label, key in pages.items():
            if st.button(label, use_container_width=True):
                st.session_state["page"] = key

        st.divider()
        st.caption("MindPulse v1.0")
        st.caption("Not a medical tool")

    # Route to correct page
    page = st.session_state.get("page", "home")

    if page == "home":
        page_home()
    elif page == "text":
        page_text_analysis(models)
    elif page == "quiz":
        page_quiz()
    elif page == "physio":
        page_physio()
    elif page == "history":
        page_history()


if __name__ == "__main__":
    main()
