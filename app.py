"""
ToxicGuard — AI-Powered Toxic Comment Detection Dashboard
=========================================================
Academic Master of Computer Applications (MCA) Mini-Project
Activity 2: Feature Extraction, Model Building, Evaluation & Dashboard Deployment.

Main Streamlit Application: app.py
"""

import os
import re
import json
import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ==============================================================================
# 1. PAGE CONFIGURATION & THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="ToxicGuard — AI-Powered Toxic Comment Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, high-contrast, modern interface
st.markdown("""
<style>
    /* Global Container styling */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1250px;
    }
    
    /* Header card */
    .tg-hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #ffffff;
        padding: 24px 28px;
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.08);
    }
    .tg-hero-card h1 {
        color: #ffffff;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .tg-hero-card p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 0px;
    }

    /* Metric Card Styling */
    .metric-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.08);
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 600;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Result Badges */
    .badge-toxic {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        color: #991b1b;
        padding: 16px 20px;
        border-radius: 12px;
        font-size: 1.35rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .badge-safe {
        background-color: #f0fdf4;
        border: 2px solid #22c55e;
        color: #166534;
        padding: 16px 20px;
        border-radius: 12px;
        font-size: 1.35rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    /* Information callouts */
    .academic-callout {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 16px 0;
        font-size: 0.92rem;
        color: #334155;
    }
    
    .oov-alert {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 0.88rem;
        color: #92400e;
    }

    /* Tag pills */
    .token-tag {
        display: inline-block;
        background-color: #e2e8f0;
        color: #1e293b;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.82rem;
        margin: 2px 4px 2px 0;
        font-family: monospace;
    }
    .token-tag-vocab {
        display: inline-block;
        background-color: #dbeafe;
        color: #1e40af;
        border: 1px solid #bfdbfe;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.82rem;
        margin: 2px 4px 2px 0;
        font-family: monospace;
        font-weight: 600;
    }
    .token-tag-oov {
        display: inline-block;
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.82rem;
        margin: 2px 4px 2px 0;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SESSION STATE MANAGEMENT
# ==============================================================================
if "single_history" not in st.session_state:
    st.session_state.single_history = []
if "batch_history" not in st.session_state:
    st.session_state.batch_history = []
if "total_session_single_analyzed" not in st.session_state:
    st.session_state.total_session_single_analyzed = 0
if "total_session_batch_analyzed" not in st.session_state:
    st.session_state.total_session_batch_analyzed = 0

# ==============================================================================
# 3. RESOURCE CACHING & ARTIFACT LOADING
# ==============================================================================
@st.cache_resource(show_spinner="Initializing NLP Resources & Linguistic Models...")
def load_nlp_pipeline():
    """
    Downloads NLTK components silently and loads:
    - Primary Model: Logistic Regression (balanced weights)
    - Baseline Model: Multinomial Naive Bayes (alpha=1.0)
    - Fitted TF-IDF Vectorizer
    """
    # 1. Download required NLTK corpora
    nltk_resources = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
    for r in nltk_resources:
        try:
            nltk.download(r, quiet=True)
        except Exception:
            pass

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    # 2. Path resolution for models
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    lr_path = os.path.join(models_dir, "toxic_comment_model.pkl")
    nb_path = os.path.join(models_dir, "naive_bayes_model.pkl")
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")

    # Verify model artifact presence
    missing_files = []
    if not os.path.exists(lr_path):
        missing_files.append("toxic_comment_model.pkl")
    if not os.path.exists(vec_path):
        missing_files.append("tfidf_vectorizer.pkl")
    
    if missing_files:
        raise FileNotFoundError(
            f"Missing required model artifacts in 'models/': {', '.join(missing_files)}. "
            f"Please run 'python src/train_model.py' to generate and serialize them."
        )

    # Load artifacts using Joblib
    lr_model = joblib.load(lr_path)
    tfidf = joblib.load(vec_path)

    nb_model = None
    if os.path.exists(nb_path):
        nb_model = joblib.load(nb_path)

    return {
        "lr_model": lr_model,
        "nb_model": nb_model,
        "tfidf": tfidf,
        "stop_words": stop_words,
        "lemmatizer": lemmatizer,
        "vocab": set(tfidf.get_feature_names_out()),
        "vocab_size": len(tfidf.get_feature_names_out())
    }

@st.cache_data(show_spinner="Loading Verified Project Dataset...")
def load_project_dataset():
    """
    Loads Activity 1 & 2 verified ground-truth dataset (toxic_comments_200_annotated.csv).
    Returns cleaned dataframe with actual label distributions.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "toxic_comments_200_annotated.csv")
    
    if not os.path.exists(csv_path):
        alt_path = os.path.join(base_dir, "..", "toxic_comments_200_annotated.csv")
        if os.path.exists(alt_path):
            csv_path = alt_path
        else:
            return None, None

    df = pd.read_csv(csv_path)
    
    # Ground truth mapping
    if "manual_label" in df.columns:
        label_col = "manual_label"
        toxic_count = int((df[label_col] == "Toxic").sum())
        non_toxic_count = int((df[label_col] == "Non-Toxic").sum())
    elif "toxic" in df.columns:
        label_col = "toxic"
        toxic_count = int((df[label_col] == 1).sum())
        non_toxic_count = int((df[label_col] == 0).sum())
    else:
        label_col = None
        toxic_count = 0
        non_toxic_count = 0

    return df, {
        "total_records": len(df),
        "toxic_count": toxic_count,
        "non_toxic_count": non_toxic_count,
        "toxic_pct": (toxic_count / len(df) * 100) if len(df) > 0 else 0,
        "non_toxic_pct": (non_toxic_count / len(df) * 100) if len(df) > 0 else 0,
        "label_col": label_col
    }

@st.cache_data(show_spinner="Loading Verified Model Evaluation Metrics...")
def load_evaluation_metrics():
    """
    Loads the Activity 2 test evaluation results from results/model_evaluation_metrics.json.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.join(base_dir, "results", "model_evaluation_metrics.json")
    
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

# ==============================================================================
# 4. TEXT PREPROCESSING PIPELINE (STRICTLY IDENTICAL TO TRAINING)
# ==============================================================================
def clean_text(text):
    """
    Normalization pipeline identical to Activity 1 & 2:
    - Lowercasing
    - Strip HTML tags
    - Strip URLs
    - Replace newlines and tabs
    - Collapse repeated whitespace
    """
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[\n\t\r]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_comment(text, stop_words, lemmatizer):
    """
    Complete NLP pipeline: Cleaning -> Tokenization -> Stopword removal -> Lemmatization.
    """
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    filtered = [w for w in tokens if w not in stop_words and len(w) > 1]
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered]
    return " ".join(lemmatized), lemmatized

def predict_single_comment(comment, model, tfidf, stop_words, lemmatizer, vocab):
    """
    Runs inference for a single comment and returns prediction, probability,
    and linguistic transparency diagnostics.
    """
    processed_str, token_list = preprocess_comment(comment, stop_words, lemmatizer)
    
    # Check vocabulary matches
    in_vocab = [w for w in token_list if w in vocab]
    oov = [w for w in token_list if w not in vocab]
    
    # Feature vectorization strictly using fitted TF-IDF
    tfidf_vec = tfidf.transform([processed_str])
    
    # Classification
    pred_idx = int(model.predict(tfidf_vec)[0])
    
    # Probability estimation
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(tfidf_vec)[0]
        prob_toxic = float(probs[1])
        prob_non_toxic = float(probs[0])
        confidence = float(probs[pred_idx])
    else:
        prob_toxic = 1.0 if pred_idx == 1 else 0.0
        prob_non_toxic = 1.0 - prob_toxic
        confidence = 1.0

    label_str = "Toxic" if pred_idx == 1 else "Non-Toxic"
    
    return {
        "original_comment": comment,
        "processed_str": processed_str,
        "tokens": token_list,
        "in_vocab": in_vocab,
        "oov": oov,
        "is_all_oov": (len(in_vocab) == 0 and len(token_list) > 0),
        "is_empty_tokens": (len(token_list) == 0),
        "prediction": label_str,
        "confidence": confidence,
        "prob_toxic": prob_toxic,
        "prob_non_toxic": prob_non_toxic,
        "vector_nonzeros": int(tfidf_vec.nnz)
    }

# ==============================================================================
# 5. INITIALIZE APPLICATION DATA & MODELS
# ==============================================================================
try:
    nlp_artifacts = load_nlp_pipeline()
    pipeline_loaded = True
except Exception as e:
    nlp_artifacts = None
    pipeline_loaded = False
    pipeline_error_msg = str(e)

df_dataset, dataset_stats = load_project_dataset()
eval_metrics = load_evaluation_metrics()

# ==============================================================================
# 6. SIDEBAR CONTROLS & NAVIGATION
# ==============================================================================
with st.sidebar:
    st.markdown("## 🛡️ ToxicGuard")
    st.markdown("**AI-Powered Toxic Comment Detection**")
    st.caption("Master of Computer Applications (MCA) Mini-Project")
    st.markdown("---")
    
    # Navigation Radio
    navigation_option = st.radio(
        "Navigation",
        options=[
            "📊 Overview",
            "💬 Single Comment Detection",
            "📁 Batch Comment Detection",
            "📈 Dataset Analytics",
            "🎯 Model Performance",
            "ℹ️ About the Project"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    
    # Active Model Selection
    if pipeline_loaded:
        available_models = ["Logistic Regression (Selected - Balanced)"]
        if nlp_artifacts["nb_model"] is not None:
            available_models.append("Multinomial Naive Bayes (Laplace alpha=1.0)")
            
        selected_model_label = st.selectbox(
            "Active Classification Model:",
            options=available_models,
            index=0,
            help="Select which trained classifier to use for real-time inference."
        )
        
        if "Logistic Regression" in selected_model_label:
            active_model = nlp_artifacts["lr_model"]
            active_model_name = "Logistic Regression"
        else:
            active_model = nlp_artifacts["nb_model"]
            active_model_name = "Multinomial Naive Bayes"
            
        st.info(f"**Selected:** {active_model_name}\n\n**TF-IDF Features:** {nlp_artifacts['vocab_size']}\n\n**N-Gram Range:** (1, 2)")
    else:
        st.error("⚠️ Model artifacts missing!")
        active_model = None
        active_model_name = "None"
        
    st.markdown("---")
    st.markdown("### 📌 Viva Quick Links")
    st.caption(
        "- **Curated Dataset:** 200 Wiki Talk Comments\n"
        "- **Test Split:** 40 comments (80:20 stratified)\n"
        "- **LR Accuracy:** 72.50% | **F1:** 75.56%\n"
        "- **NB Accuracy:** 67.50% | **F1:** 73.47%"
    )
    st.markdown("---")
    st.caption("Developed for Academic NLP Activity 2 Evaluation.")

# ==============================================================================
# 7. SECTION ROUTING & RENDERING
# ==============================================================================

# ------------------------------------------------------------------------------
# SECTION 1: OVERVIEW
# ------------------------------------------------------------------------------
if navigation_option == "📊 Overview":
    st.markdown("""
    <div class="tg-hero-card">
        <h1>🛡️ ToxicGuard — Toxic Comment Detection</h1>
        <p>An Academic Natural Language Processing and Supervised Machine Learning Dashboard for automated toxicity detection, feature transparency, batch moderation, and performance diagnostics.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📌 Ground-Truth Dataset Summary (Activity 1 & 2)")
    st.markdown("Actual metrics computed directly from the curated project dataset (`toxic_comments_200_annotated.csv`).")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{dataset_stats['total_records'] if dataset_stats else '200'}</div>
            <div class="metric-label">Total Comments</div>
            <div class="metric-sub">Ground-Truth Dataset</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: #dc2626;">{dataset_stats['toxic_count'] if dataset_stats else '107'}</div>
            <div class="metric-label">Toxic Comments</div>
            <div class="metric-sub">{dataset_stats['toxic_pct'] if dataset_stats else 53.5:.1f}% Ground Truth</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: #16a34a;">{dataset_stats['non_toxic_count'] if dataset_stats else '93'}</div>
            <div class="metric-label">Non-Toxic Comments</div>
            <div class="metric-sub">{dataset_stats['non_toxic_pct'] if dataset_stats else 46.5:.1f}% Ground Truth</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: #2563eb;">{active_model_name}</div>
            <div class="metric-label">Active Model</div>
            <div class="metric-sub">Vocab: {nlp_artifacts['vocab_size'] if nlp_artifacts else '764'} N-Grams</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Session Activity Widget (Distinguished from dataset records)
    st.subheader("⏱️ Current Session Activity")
    st.markdown("Tracks comments analyzed during your current active browser session (isolated from historical dataset records).")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric("Single Comments Analyzed This Session", st.session_state.total_session_single_analyzed)
    with sc2:
        st.metric("Batch Records Processed This Session", st.session_state.total_session_batch_analyzed)
    with sc3:
        st.metric("Test Partition Verification Accuracy", "72.50%", delta="Selected Model (LR)")

    st.markdown("---")

    # Workflow Architecture
    st.subheader("🧩 End-to-End NLP Pipeline Architecture")
    st.markdown("""
    ```
    [Raw Comment Text]
            │
            ▼
    [1. Normalization] ──> Lowercase, strip HTML/URLs, remove excess whitespace
            │
            ▼
    [2. Linguistic Preprocessing] ──> NLTK Word Tokenize, English Stopwords, WordNet Lemmatize
            │
            ▼
    [3. TF-IDF Vectorizer] ──> Fitted strictly on X_train (Vocabulary: 764 unigrams & bigrams)
            │
            ▼
    [4. Machine Learning Classifier] ──> Logistic Regression (Balanced) or Naive Bayes (Laplace alpha=1.0)
            │
            ▼
    [5. Decision Output] ──> Binary Predicted Class (Toxic / Non-Toxic) + Model-Estimated Probability
    ```
    """)

    # Quick Start Action Cards
    st.markdown("### 🚀 Quick Navigation")
    qcol1, qcol2 = st.columns(2)
    with qcol1:
        st.markdown("""
        <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 10px; padding: 18px;">
            <h4>💬 Test a Single Comment</h4>
            <p>Analyze individual sentences in real time, view detected vocabulary tokens, and inspect posterior probabilities.</p>
        </div>
        """, unsafe_allow_html=True)
    with qcol2:
        st.markdown("""
        <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 10px; padding: 18px;">
            <h4>📁 Batch Process a CSV File</h4>
            <p>Upload a CSV file containing multiple user comments, run automated moderation, and download enriched results.</p>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# SECTION 2: SINGLE COMMENT DETECTION
# ------------------------------------------------------------------------------
elif navigation_option == "💬 Single Comment Detection":
    st.header("💬 Single Comment Toxicity Detection")
    st.markdown("Enter custom text or choose an example comment to analyze using the selected trained classification pipeline.")

    if not pipeline_loaded:
        st.error(f"Cannot perform prediction because model files were not found: {pipeline_error_msg}")
        st.stop()

    # Preset examples for fast viva testing
    st.markdown("##### 💡 Try an Example Preset:")
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    preset_text = ""
    with pcol1:
        if st.button("Civil & Supportive", use_container_width=True):
            preset_text = "Thank you so much for your assistance, this really helped me understand the topic!"
    with pcol2:
        if st.button("Direct Insult", use_container_width=True):
            preset_text = "You are an absolute idiot and have no idea what you are talking about."
    with pcol3:
        if st.button("Hostile Threat", use_container_width=True):
            preset_text = "Shut up right now or I will make you pay for editing this page."
    with pcol4:
        if st.button("Out-of-Vocabulary (OOV)", use_container_width=True):
            preset_text = "You are completely useless."

    # Input text area
    user_input = st.text_area(
        "Enter comment text below:",
        value=preset_text if preset_text else "",
        placeholder="Type or paste a comment here to evaluate its toxicity...",
        height=130
    )

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        analyze_clicked = st.button("🔍 Analyze Comment", type="primary", use_container_width=True)

    if analyze_clicked:
        cleaned_raw = user_input.strip()
        if not cleaned_raw:
            st.warning("⚠️ Input is empty. Please enter a valid text comment before analyzing.")
        else:
            if len(cleaned_raw) > 2000:
                st.info("ℹ️ Note: This is a very long comment (>2,000 characters). Full text will be processed through the tokenizer.")

            with st.spinner("Processing text through NLP pipeline..."):
                res = predict_single_comment(
                    comment=cleaned_raw,
                    model=active_model,
                    tfidf=nlp_artifacts["tfidf"],
                    stop_words=nlp_artifacts["stop_words"],
                    lemmatizer=nlp_artifacts["lemmatizer"],
                    vocab=nlp_artifacts["vocab"]
                )

                st.session_state.total_session_single_analyzed += 1
                st.session_state.single_history.insert(0, {
                    "Comment": cleaned_raw[:70] + "..." if len(cleaned_raw) > 70 else cleaned_raw,
                    "Model": active_model_name,
                    "Prediction": res["prediction"],
                    "P(Toxic)": f"{res['prob_toxic']*100:.2f}%",
                    "P(Non-Toxic)": f"{res['prob_non_toxic']*100:.2f}%"
                })

            st.markdown("### 🎯 Classification Result")
            
            if res["prediction"] == "Toxic":
                st.markdown(f"""
                <div class="badge-toxic">
                    <span>🚨</span>
                    <div>
                        <div>PREDICTED CLASS: <strong>TOXIC</strong></div>
                        <div style="font-size: 0.95rem; font-weight: 500; color: #7f1d1d; margin-top: 2px;">
                            Model-Estimated Toxicity Probability: <strong>{res['prob_toxic']*100:.2f}%</strong>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="badge-safe">
                    <span>✅</span>
                    <div>
                        <div>PREDICTED CLASS: <strong>NON-TOXIC (SAFE)</strong></div>
                        <div style="font-size: 0.95rem; font-weight: 500; color: #14532d; margin-top: 2px;">
                            Model-Estimated Non-Toxicity Probability: <strong>{res['prob_non_toxic']*100:.2f}%</strong> (P(Toxic)={res['prob_toxic']*100:.2f}%)
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.caption("⚠️ **Note:** Displayed probability is a model-estimated likelihood derived from classifier weights, not an infallible guarantee of factual truth.")

            # Probability Breakdown Progress Bars
            st.markdown("#### 📊 Probability Distribution")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.write(f"**P(Non-Toxic):** {res['prob_non_toxic']*100:.2f}%")
                st.progress(res["prob_non_toxic"])
            with p_col2:
                st.write(f"**P(Toxic):** {res['prob_toxic']*100:.2f}%")
                st.progress(res["prob_toxic"])

            # Linguistic Diagnostics Expander
            st.markdown("#### 🔬 NLP Preprocessing Diagnostics & Feature Transparency")
            with st.expander("Inspect Tokenization, Vocabulary Overlap & TF-IDF Non-Zeros", expanded=True):
                st.markdown(f"**Cleaned & Lemmatized Tokens:** `{res['processed_str'] if res['processed_str'] else '[No tokens remaining]'}`")
                
                if res["in_vocab"]:
                    vocab_tags = " ".join([f"<span class='token-tag-vocab'>✓ {w}</span>" for w in res["in_vocab"]])
                    st.markdown(f"**Active Vocabulary Features Recognized ({len(res['in_vocab'])}):** {vocab_tags}", unsafe_allow_html=True)
                else:
                    st.markdown("**Active Vocabulary Features Recognized:** *None*")

                if res["oov"]:
                    oov_tags = " ".join([f"<span class='token-tag-oov'>✗ {w}</span>" for w in res["oov"]])
                    st.markdown(f"**Out-of-Vocabulary (OOV) / Filtered Tokens ({len(res['oov'])}):** {oov_tags}", unsafe_allow_html=True)

                st.markdown(f"**Non-Zero TF-IDF Matrix Entries:** `{res['vector_nonzeros']}` out of {nlp_artifacts['vocab_size']}")

                if res["is_all_oov"] or res["vector_nonzeros"] == 0:
                    st.markdown("""
                    <div class="oov-alert">
                        <strong>⚠️ Out-of-Vocabulary (OOV) Notice:</strong><br>
                        None of the tokens in this comment were present in the training vocabulary. Under TF-IDF feature representation, an unseen word vector produces all zeros: $\\vec{x} = \\mathbf{0}$.<br>
                        Consequently, the classifier defaults to its intercept bias: $\\sigma(w^T \\mathbf{0} + b) = \\sigma(b) \\approx 52.6\\%$. This mathematically explains borderline predictions for unobserved words.
                    </div>
                    """, unsafe_allow_html=True)

    # Session Analysis History
    if st.session_state.single_history:
        st.markdown("---")
        st.markdown("##### 📜 Recent Analyses in Current Session")
        history_df = pd.DataFrame(st.session_state.single_history)
        st.dataframe(history_df, use_container_width=True)
        if st.button("🗑️ Clear Session History"):
            st.session_state.single_history = []
            st.rerun()

# ------------------------------------------------------------------------------
# SECTION 3: BATCH COMMENT DETECTION
# ------------------------------------------------------------------------------
elif navigation_option == "📁 Batch Comment Detection":
    st.header("📁 Batch Comment Moderation via CSV Upload")
    st.markdown("Upload a CSV file containing user comments to execute bulk toxicity classification, generate batch analytics, and export enriched results.")

    if not pipeline_loaded:
        st.error(f"Cannot perform batch prediction because model files were not found: {pipeline_error_msg}")
        st.stop()

    st.markdown("##### 📥 Don't have a CSV file ready? Download a test sample:")
    sample_comments_df = pd.DataFrame({
        "comment_id": [101, 102, 103, 104, 105, 106, 107],
        "comment_text": [
            "Thank you for reviewing my changes, I appreciate your time.",
            "You are a complete idiot and should delete your account.",
            "I disagree with this decision, but let's review the policy.",
            "Shut your mouth or I will hunt you down and destroy you.",
            "Great contribution to the Wikipedia article!",
            "",  # Empty comment test
            "This whole website is completely useless and ruined."
        ],
        "source": ["TalkPage", "Forum", "TalkPage", "UserTalk", "TalkPage", "Forum", "UserTalk"]
    })
    
    sample_csv_bytes = sample_comments_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📄 Download Sample Comments CSV",
        data=sample_csv_bytes,
        file_name="toxicguard_sample_batch.csv",
        mime="text/csv"
    )

    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload CSV File:", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded `{uploaded_file.name}` ({len(batch_df)} rows, {len(batch_df.columns)} columns)")
            
            col_candidates = [c for c in batch_df.columns if any(k in c.lower() for k in ["comment", "text", "message", "body", "content"])]
            default_col_idx = 0
            if col_candidates:
                default_col_idx = list(batch_df.columns).index(col_candidates[0])

            selected_col = st.selectbox(
                "Select the column containing comment text:",
                options=list(batch_df.columns),
                index=default_col_idx,
                help="Select which column contains the text to be evaluated."
            )

            if st.button("🚀 Classify Batch Comments", type="primary"):
                with st.spinner("Classifying all comments in batch..."):
                    results_df = batch_df.copy()
                    raw_comments = results_df[selected_col].fillna("").astype(str).tolist()
                    
                    preds = []
                    probs_toxic = []
                    probs_non_toxic = []
                    valid_flags = []

                    for comment in raw_comments:
                        cleaned = comment.strip()
                        if not cleaned:
                            preds.append("Empty / Invalid")
                            probs_toxic.append(0.0)
                            probs_non_toxic.append(1.0)
                            valid_flags.append(False)
                        else:
                            res = predict_single_comment(
                                comment=cleaned,
                                model=active_model,
                                tfidf=nlp_artifacts["tfidf"],
                                stop_words=nlp_artifacts["stop_words"],
                                lemmatizer=nlp_artifacts["lemmatizer"],
                                vocab=nlp_artifacts["vocab"]
                            )
                            preds.append(res["prediction"])
                            probs_toxic.append(round(res["prob_toxic"], 4))
                            probs_non_toxic.append(round(res["prob_non_toxic"], 4))
                            valid_flags.append(True)

                    results_df["Predicted_Class"] = preds
                    results_df["Probability_Toxic"] = probs_toxic
                    results_df["Probability_Non_Toxic"] = probs_non_toxic
                    results_df["Is_Valid_Text"] = valid_flags

                    st.session_state.total_session_batch_analyzed += len(results_df)

                    st.markdown("### 📊 Batch Moderation Summary")
                    total_rows = len(results_df)
                    valid_count = sum(valid_flags)
                    toxic_count = sum(1 for p in preds if p == "Toxic")
                    non_toxic_count = sum(1 for p in preds if p == "Non-Toxic")
                    empty_count = total_rows - valid_count

                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Total Rows", total_rows)
                    with m2:
                        st.metric("Toxic Flagged", toxic_count, delta=f"{toxic_count/valid_count*100:.1f}%" if valid_count>0 else "0%")
                    with m3:
                        st.metric("Safe / Non-Toxic", non_toxic_count, delta=f"{non_toxic_count/valid_count*100:.1f}%" if valid_count>0 else "0%")
                    with m4:
                        st.metric("Empty / Skipped", empty_count)

                    if valid_count > 0:
                        st.markdown("#### Distribution of Batch Predictions")
                        chart_data = pd.DataFrame({
                            "Class": ["Toxic", "Non-Toxic"],
                            "Count": [toxic_count, non_toxic_count]
                        })
                        st.bar_chart(chart_data.set_index("Class"), color="#ef4444")

                    st.markdown("#### 📋 Classified Batch Data")
                    filter_choice = st.radio(
                        "Filter View:",
                        options=["All Records", "Toxic Only", "Non-Toxic Only", "Empty/Skipped Only"],
                        horizontal=True
                    )

                    if filter_choice == "Toxic Only":
                        view_df = results_df[results_df["Predicted_Class"] == "Toxic"]
                    elif filter_choice == "Non-Toxic Only":
                        view_df = results_df[results_df["Predicted_Class"] == "Non-Toxic"]
                    elif filter_choice == "Empty/Skipped Only":
                        view_df = results_df[~results_df["Is_Valid_Text"]]
                    else:
                        view_df = results_df

                    st.dataframe(view_df, use_container_width=True)

                    csv_export = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="💾 Download Classified Results as CSV",
                        data=csv_export,
                        file_name=f"classified_{uploaded_file.name}",
                        mime="text/csv",
                        type="primary"
                    )

        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")

# ------------------------------------------------------------------------------
# SECTION 4: DATASET ANALYTICS
# ------------------------------------------------------------------------------
elif navigation_option == "📈 Dataset Analytics":
    st.header("📈 Dataset Analytics & Exploratory Data Analysis")
    st.markdown("Comprehensive statistical breakdown of the verified human-annotated dataset (`toxic_comments_200_annotated.csv`).")

    if df_dataset is None:
        st.warning("Dataset file `toxic_comments_200_annotated.csv` was not found.")
        st.stop()

    st.markdown("""
    <div class="academic-callout">
        <strong>Ground-Truth Label Distinction:</strong><br>
        All metrics and charts below represent <em>actual human-verified ground-truth annotations</em> from Activity 1. These are historical labeled records, completely distinct from real-time model predictions.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("1. Ground-Truth Class Distribution")
        dist_df = pd.DataFrame({
            "Class": ["Toxic (1)", "Non-Toxic (0)"],
            "Count": [dataset_stats["toxic_count"], dataset_stats["non_toxic_count"]],
            "Percentage": [f"{dataset_stats['toxic_pct']:.1f}%", f"{dataset_stats['non_toxic_pct']:.1f}%"]
        })
        st.table(dist_df)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x="Class", y="Count", data=dist_df, palette=["#ef4444", "#22c55e"], ax=ax)
        ax.set_title("Ground-Truth Toxicity Distribution (N=200)", fontweight="bold")
        ax.set_ylabel("Number of Comments")
        for i, row in dist_df.iterrows():
            ax.text(i, row["Count"] + 2, f"{row['Count']} ({row['Percentage']})", ha="center", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        st.subheader("2. Toxicity Subcategory Breakdown")
        if "category" in df_dataset.columns:
            cat_counts = df_dataset["category"].value_counts().reset_index()
            cat_counts.columns = ["Subcategory", "Count"]
            
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.barplot(y="Subcategory", x="Count", data=cat_counts, palette="magma", ax=ax2)
            ax2.set_title("Toxicity Subcategories (Human Annotated)", fontweight="bold")
            ax2.set_xlabel("Count")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()
        else:
            st.info("Subcategory column 'category' not found in dataset.")

    st.markdown("---")
    st.subheader("3. Comment Length Distribution")
    
    df_calc = df_dataset.copy()
    df_calc["char_length"] = df_calc["comment_text"].astype(str).apply(len)
    df_calc["word_count"] = df_calc["comment_text"].astype(str).apply(lambda s: len(s.split()))

    col_l1, col_l2 = st.columns(2)
    with col_l1:
        fig_l, ax_l = plt.subplots(figsize=(6, 3.5))
        sns.histplot(data=df_calc, x="word_count", hue=dataset_stats["label_col"], bins=30, kde=True, ax=ax_l, palette=["#22c55e", "#ef4444"])
        ax_l.set_title("Word Count Distribution by Class", fontweight="bold")
        ax_l.set_xlabel("Word Count")
        plt.tight_layout()
        st.pyplot(fig_l)
        plt.close()

    with col_l2:
        st.markdown("##### Summary Statistics by Class:")
        grouped = df_calc.groupby(dataset_stats["label_col"])[["word_count", "char_length"]].agg(["mean", "median", "std"]).round(1)
        st.dataframe(grouped, use_container_width=True)
        st.caption("Observations indicate that toxic comments frequently exhibit shorter, punchy lengths with aggressive insults or profanity.")

    st.markdown("---")
    st.subheader("4. Annotated Dataset Explorer")
    search_term = st.text_input("🔍 Search comments by keyword:", "")
    filtered_df = df_dataset.copy()
    if search_term:
        filtered_df = filtered_df[filtered_df["comment_text"].str.contains(search_term, case=False, na=False)]
    
    st.dataframe(filtered_df[["comment_text", dataset_stats["label_col"]] + ([c for c in ["category"] if c in df_dataset.columns])].head(25), use_container_width=True)

# ------------------------------------------------------------------------------
# SECTION 5: MODEL PERFORMANCE
# ------------------------------------------------------------------------------
elif navigation_option == "🎯 Model Performance":
    st.header("🎯 Model Performance & Evaluation Benchmarking")
    st.markdown("Verified quantitative metrics evaluated on the independent test set ($N=40$ comments, stratified 80:20 partition).")

    if eval_metrics is None:
        st.warning("Saved evaluation metrics (`results/model_evaluation_metrics.json`) not found.")
        st.stop()

    st.markdown("""
    <div class="academic-callout">
        <strong>Strict Data Leakage Prevention Protocol:</strong><br>
        The 80:20 stratified train-test split was executed <em>prior</em> to TF-IDF feature fitting. The TF-IDF vectorizer was fitted strictly on the 160 training observations, ensuring that the 40 test observations remained completely unseen until final evaluation.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("1. Quantitative Model Comparison (Test Split N=40)")
    
    lr_data = eval_metrics["models"]["Logistic Regression"]
    nb_data = eval_metrics["models"]["Multinomial Naive Bayes"]

    comparison_df = pd.DataFrame([
        {
            "Metric": "Accuracy",
            "Logistic Regression (Selected)": f"{lr_data['accuracy']*100:.2f}%",
            "Multinomial Naive Bayes": f"{nb_data['accuracy']*100:.2f}%",
            "Advantage": "Logistic Regression (+5.00%)"
        },
        {
            "Metric": "Toxic Precision",
            "Logistic Regression (Selected)": f"{lr_data['toxic_precision']*100:.2f}%",
            "Multinomial Naive Bayes": f"{nb_data['toxic_precision']*100:.2f}%",
            "Advantage": "Logistic Regression (+6.54%)"
        },
        {
            "Metric": "Toxic Recall",
            "Logistic Regression (Selected)": f"{lr_data['toxic_recall']*100:.2f}%",
            "Multinomial Naive Bayes": f"{nb_data['toxic_recall']*100:.2f}%",
            "Advantage": "Multinomial Naive Bayes (+4.76%)"
        },
        {
            "Metric": "Toxic F1-Score",
            "Logistic Regression (Selected)": f"{lr_data['toxic_f1']*100:.2f}%",
            "Multinomial Naive Bayes": f"{nb_data['toxic_f1']*100:.2f}%",
            "Advantage": "Logistic Regression (+2.09%)"
        },
        {
            "Metric": "Macro F1-Score",
            "Logistic Regression (Selected)": f"{lr_data['macro_f1']*100:.2f}%",
            "Multinomial Naive Bayes": f"{nb_data['macro_f1']*100:.2f}%",
            "Advantage": "Logistic Regression (+6.29%)"
        },
        {
            "Metric": "False Positives (Type I)",
            "Logistic Regression (Selected)": f"{lr_data['fp']} / 19 Non-Toxic",
            "Multinomial Naive Bayes": f"{nb_data['fp']} / 19 Non-Toxic",
            "Advantage": "Logistic Regression (3 Fewer False Flags)"
        },
        {
            "Metric": "False Negatives (Type II)",
            "Logistic Regression (Selected)": f"{lr_data['fn']} / 21 Toxic",
            "Multinomial Naive Bayes": f"{nb_data['fn']} / 21 Toxic",
            "Advantage": "Multinomial Naive Bayes (1 Fewer Miss)"
        }
    ])
    st.table(comparison_df.set_index("Metric"))

    st.markdown("---")
    
    st.subheader("2. Confusion Matrix Heatmaps")
    cm_col1, cm_col2 = st.columns(2)
    
    with cm_col1:
        st.markdown("##### Logistic Regression (Primary)")
        cm_lr = np.array(lr_data["confusion_matrix"])
        fig_lr, ax_lr = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_lr,
                    xticklabels=["Non-Toxic (0)", "Toxic (1)"], yticklabels=["Non-Toxic (0)", "Toxic (1)"],
                    annot_kws={"size": 14, "weight": "bold"})
        ax_lr.set_xlabel("Predicted Label")
        ax_lr.set_ylabel("Actual Ground Truth")
        ax_lr.set_title(f"Acc: {lr_data['accuracy']*100:.1f}%, F1: {lr_data['toxic_f1']*100:.1f}%", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_lr)
        plt.close()

    with cm_col2:
        st.markdown("##### Multinomial Naive Bayes")
        cm_nb = np.array(nb_data["confusion_matrix"])
        fig_nb, ax_nb = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Oranges", cbar=False, ax=ax_nb,
                    xticklabels=["Non-Toxic (0)", "Toxic (1)"], yticklabels=["Non-Toxic (0)", "Toxic (1)"],
                    annot_kws={"size": 14, "weight": "bold"})
        ax_nb.set_xlabel("Predicted Label")
        ax_nb.set_ylabel("Actual Ground Truth")
        ax_nb.set_title(f"Acc: {nb_data['accuracy']*100:.1f}%, F1: {nb_data['toxic_f1']*100:.1f}%", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_nb)
        plt.close()

    st.markdown("---")

    st.subheader("3. Detailed Classification Reports")
    rcol1, rcol2 = st.columns(2)
    with rcol1:
        st.markdown("**Logistic Regression Report:**")
        cr_lr = pd.DataFrame(lr_data["classification_report"]).transpose().round(2)
        st.dataframe(cr_lr, use_container_width=True)
    with rcol2:
        st.markdown("**Multinomial Naive Bayes Report:**")
        cr_nb = pd.DataFrame(nb_data["classification_report"]).transpose().round(2)
        st.dataframe(cr_nb, use_container_width=True)

    st.markdown("---")

    st.subheader("4. Selected Model Justification")
    st.markdown(f"""
    <div style="background-color: #f1f5f9; border-left: 5px solid #2563eb; padding: 18px; border-radius: 8px;">
        <h4 style="margin-top: 0; color: #1e3a8a;">Why Logistic Regression is Selected:</h4>
        <p style="margin-bottom: 8px;">
            In automated moderation systems, <strong>False Positives (Type I errors)</strong> carry severe costs: they wrongly silence innocent users, suppress legitimate discourse, and damage platform credibility.
        </p>
        <ul style="margin-bottom: 0;">
            <li><strong>Precision Advantage:</strong> Logistic Regression achieved <strong>70.83%</strong> toxic precision versus <strong>64.29%</strong> for Naive Bayes.</li>
            <li><strong>Lower False Positive Rate:</strong> Naive Bayes generated <strong>10 false positives</strong> out of 19 safe comments (a false alarm rate exceeding 50%), whereas Logistic Regression reduced this to 7.</li>
            <li><strong>Superior F1-Score:</strong> Overall harmonic mean F1-Score for the Toxic class reached <strong>75.56%</strong> (LR) vs <strong>73.47%</strong> (NB).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# SECTION 6: ABOUT THE PROJECT
# ------------------------------------------------------------------------------
elif navigation_option == "ℹ️ About the Project":
    st.header("ℹ️ About ToxicGuard & Technical Methodology")
    st.markdown("Comprehensive academic documentation of NLP concepts, mathematical representations, algorithmic choices, and operational limitations.")

    with st.expander("1. What is Toxic Comment Detection & Why is NLP Essential?", expanded=True):
        st.markdown("""
        **Online Toxicity Problem:**
        User-generated forums, talk pages, and community platforms experience a relentless inflow of comments. Human moderation cannot scale to millions of messages and causes severe psychological fatigue for human moderators.
        
        **Why Rule-Based Blacklists Fail:**
        Early profanity filters relied on exact string matches against a blacklist of swear words. These fail because:
        1. They miss obfuscated spellings (*'fck'*, *'b!tch'*).
        2. They cannot evaluate contextual intent (e.g., medical anatomy discussions vs abusive harassment).
        3. They produce massive false positive rates on benign comments.
        
        Natural Language Processing (NLP) converts unstructured text into statistical vector representations that reflect empirical semantic probabilities rather than rigid keywords.
        """)

    with st.expander("2. Mathematical Formulation: How TF-IDF Works", expanded=True):
        st.markdown(r"""
        **Term Frequency - Inverse Document Frequency (TF-IDF)** scales the importance of a word $t$ in a document $d$ relative to a corpus $D$:
        
        $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
        
        Where:
        - **Sublinear Term Frequency:** $\text{TF}(t, d) = 1 + \log(\text{count}(t, d))$ (prevents high-frequency words from disproportionately dominating).
        - **Smooth Inverse Document Frequency:**
          $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + \text{DF}(t, D)}\right) + 1$$
        - **N-Gram Range $(1, 2)$:** Captures both isolated unigrams (e.g., *"idiot"*) and contiguous bigrams (e.g., *"shut up"*), preserving critical phrase-level context.
        """)

    with st.expander("3. Machine Learning Algorithms Evaluated", expanded=True):
        st.markdown(r"""
        **Model 1: Logistic Regression (Selected Primary Model)**
        - Optimizes binary cross-entropy log-loss:
          $$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
        - Utilizes `class_weight='balanced'` to inversely weight class frequencies, mitigating dataset skew.
        
        **Model 2: Multinomial Naive Bayes (Baseline Model)**
        - Computes posterior class likelihoods using Bayes' theorem under the conditional feature independence assumption:
          $$P(y=c|\mathbf{x}) \propto P(y=c) \prod_{j=1}^{V} P(x_j|y=c)$$
        - Incorporates **Laplace Smoothing** ($\alpha=1.0$) to avoid zero-probability estimates for unobserved feature occurrences:
          $$\hat{\theta}_{ci} = \frac{N_{ci} + \alpha}{N_c + \alpha V}$$
        """)

    with st.expander("4. System Limitations & Academic Insights", expanded=True):
        st.markdown("""
        1. **Sample Size Constraints:** The curated academic dataset contains 200 human-annotated comments ($N=160$ train, $N=40$ test). In a 40-comment test split, each individual misclassification shifts the accuracy metric by 2.5%.
        2. **Out-of-Vocabulary (OOV) Invariance:** Traditional TF-IDF models cannot infer representations for words absent from the training set. If all words in an input sentence are OOV, the TF-IDF vector is zero and the model defaults to its baseline intercept.
        3. **Absence of Sarcasm & Deep Syntactic Understanding:** Linear bag-of-words classifiers evaluate word co-occurrence without modeling syntactic dependency trees or sarcasm.
        4. **Binary Simplification:** Real-world toxicity spans varied dimensions (threats, hate speech, obscenity, trolling). Binary classification collapses these nuances into a single score.
        """)

    with st.expander("5. Academic Credits & Project Information", expanded=False):
        st.markdown("""
        - **Project Name:** ToxicGuard — AI-Powered Toxic Comment Detection Dashboard
        - **Course:** Master of Computer Applications (MCA) Mini-Project
        - **Milestone:** Activity 2 — Supervised NLP Classification & Streamlit Deployment
        - **Core Libraries:** Python 3.13, Scikit-Learn, Streamlit, NLTK, Pandas, NumPy, Matplotlib, Seaborn, Joblib.
        - **Dataset Reference:** Wikipedia Toxic Comment Classification Challenge (Kaggle).
        """)

# ==============================================================================
# 8. FOOTER
# ==============================================================================
st.markdown("---")
f_col1, f_col2 = st.columns([3, 1])
with f_col1:
    st.caption("🛡️ **ToxicGuard Dashboard** | Master of Computer Applications (MCA) NLP Mini-Project Activity 2.")
with f_col2:
    st.caption("Active Model: **" + active_model_name + "**")
