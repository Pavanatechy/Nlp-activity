# TOXICGUARD: AI-POWERED TOXIC COMMENT DETECTION
## Final Project Presentation Slides (MCA Mini-Project Defense & Dashboard Demonstration)

---

### SLIDE 1: Title & Project Overview
- **Project Title:** ToxicGuard: AI-Powered Toxic Comment Detection Dashboard
- **Academic Program:** Master of Computer Applications (MCA) Mini-Project
- **Phase:** Activity 2 & Dashboard Phase — Feature Extraction, Model Building, Evaluation & Streamlit Deployment
- **Candidate:** MCA Student
- **Mentor / Guide:** Department of Computer Applications
- **Domain:** Natural Language Processing (NLP) | Supervised Machine Learning | Web Engineering

> **Speaker Notes:**  
> "Good morning respected evaluators and faculty members. Today, I am presenting my MCA mini-project titled 'ToxicGuard: AI-Powered Toxic Comment Detection Dashboard'. In this phase, we completed an end-to-end NLP and machine learning pipeline, rigorously benchmarked classification models on held-out test data, and deployed a full-featured local Streamlit web application for real-time and batch moderation."

---

### SLIDE 2: Problem Statement & Objectives
- **Problem Statement:**  
  - Rapid escalation of toxic, abusive, threatening, and obscene commentary on user platforms and collaborative forums.
  - Manual human moderation is intractable due to sheer content velocity and psychological fatigue.
- **Project Objectives:**  
  1. Build a robust NLP text normalization and lemmatization pipeline.
  2. Implement strict **data leakage prevention** via pre-vectorization stratified train-test partitioning.
  3. Extract unigram and bigram **TF-IDF features** strictly on training data.
  4. Train, benchmark, and evaluate **Logistic Regression (Balanced)** vs. **Multinomial Naive Bayes (Laplace Smoothed)**.
  5. Deploy **ToxicGuard (`app.py`)**, a local Streamlit web dashboard providing single comment inference, batch CSV moderation, dataset analytics, and model performance visualization.

> **Speaker Notes:**  
> "Our goal was not just to train an offline script, but to deliver a functional, transparent, and mathematically rigorous content moderation system complete with confidence estimation and batch CSV processing."

---

### SLIDE 3: Literature & Technical Background
- **Lexicon-Based Moderation:** Rule-based blacklists and keyword filters fail on context, miss obfuscated spellings, and generate high false positives.
- **Vector Space Modeling:** Term Frequency - Inverse Document Frequency (Salton & Buckley, 1988) scales term importance by down-weighting ubiquitously occurring words.
- **Probabilistic vs. Discriminative Classifiers:**
  - *Multinomial Naive Bayes:* Rapid probabilistic classifier under conditional feature independence; strong text baseline.
  - *Logistic Regression:* Discriminative linear classifier optimizing log-loss; offers calibrated posterior probabilities and balanced class weights.
- **Transformer Context:** Modern deep transformers (BERT) require heavy GPU resources; classical TF-IDF with linear classifiers represents the optimal lightweight, zero-GPU baseline.

> **Speaker Notes:**  
> "TF-IDF paired with regularized Logistic Regression provides millisecond inference times, zero GPU dependencies, and high transparency into feature weights."

---

### SLIDE 4: Dataset Description & Ground-Truth Annotation
- **Source Benchmark:** Kaggle Wikipedia Toxic Comment Classification Challenge (159,571 comments).
  - Demonstrates severe class imbalance (only 9.58% positive toxic comments).
- **Curated Activity 1 Corpus:** `toxic_comments_200_annotated.csv` ($N=200$ comments).
  - Sampled 100 original positive and 100 original negative comments for balanced manual validation.
  - Refined via human manual annotation:
    - **Toxic (1):** 107 comments (53.5%)
    - **Non-Toxic (0):** 93 comments (46.5%)
- **Toxicity Subcategories Identified:** Obscene/Profanity (45), Insult (24), Other (22), Identity Hate (7), Threat (5), Harassment (4).

> **Speaker Notes:**  
> "We used our curated 200-sample human-annotated dataset. Human verification caught subtle abusive comments that automated tags had missed, establishing a reliable ground truth."

---

### SLIDE 5: Methodology & Leakage Prevention Protocol
```
[Raw Comment Text]
        │
        ▼
[Text Cleaning & Normalization] (HTML stripping, URL removal, lowercasing)
        │
        ▼
[Tokenization & Lemmatization] (NLTK word_tokenize, WordNet lemmatizer)
        │
        ▼
[Stratified 80:20 Split] (Train N=160, Test N=40) ──> Zero Data Leakage
        │
        ▼
[TF-IDF Feature Extraction] (Fitted ONLY on Training Split; Vocab: 764 n-grams)
        │
        ├───> [Logistic Regression (Balanced Weights)]
        └───> [Multinomial Naive Bayes (Laplace alpha=1.0)]
```
- **Key Scientific Safeguard:** Vocabulary fitting occurred *strictly* on `X_train`. The test set `X_test` remained completely unseen until evaluation.

> **Speaker Notes:**  
> "A critical requirement in machine learning is preventing data leakage. By partitioning before vectorization, our test metrics reflect true generalization performance on unseen text."

---

### SLIDE 6: Experimental Evaluation Results ($N=40$ Test Split)

| Metric | Logistic Regression (Selected) | Multinomial Naive Bayes | Best Performer |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **72.50%** | 67.50% | **Logistic Regression** (+5.00%) |
| **Toxic Precision** | **70.83%** | 64.29% | **Logistic Regression** (+6.54%) |
| **Toxic Recall** | 80.95% | **85.71%** | Multinomial Naive Bayes (+4.76%) |
| **Toxic F1-Score** | **75.56%** | 73.47% | **Logistic Regression** (+2.09%) |
| **Macro F1-Score** | **72.06%** | 65.77% | **Logistic Regression** (+6.29%) |
| **True Positives (TP)** | 17 | 18 | Multinomial Naive Bayes |
| **True Negatives (TN)** | **12** | 9 | **Logistic Regression** |
| **False Positives (FP)** | **7** | 10 | **Logistic Regression (3 fewer false alarms)** |
| **False Negatives (FN)** | 4 | **3** | Multinomial Naive Bayes |

> **Speaker Notes:**  
> "All metrics presented are actual outputs evaluated on our held-out test split. Logistic Regression is our selected model because it achieves higher overall accuracy and 70.83% precision, avoiding unwarranted censorship."

---

### SLIDE 7: Selected Model Justification & Error Analysis
- **The Content Moderation Trade-Off:**
  - **Type I Error (False Positive):** Flagging a polite comment as toxic. Silences innocent users and ruins user experience.
  - **Type II Error (False Negative):** Missing a toxic comment.
- **Why Logistic Regression Won:**
  - Multinomial Naive Bayes produced **10 false positives** out of 19 non-toxic comments (>52% false alarm rate).
  - Logistic Regression reduced false positives to **7**, delivering **70.83% precision** and an **F1-Score of 75.56%**.
- **Out-of-Vocabulary (OOV) Handling:**
  - Words not seen during training produce zero vectors $\mathbf{x}=\mathbf{0}$.
  - The model defaults gracefully to its intercept: $\sigma(w^T \mathbf{0} + b) = \sigma(0.1035) \approx 52.6\%$.

> **Speaker Notes:**  
> "In moderation, false accusations are fatal to community trust. Logistic Regression provides the highest precision and a superior F1-score while remaining transparent about OOV limitations."

---

### SLIDE 8: ToxicGuard Web Dashboard Architecture (`app.py`)
```
                                 [ToxicGuard Sidebar]
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
             [Model Switcher]                            [Navigation Menu]
      - Logistic Regression (Primary)              1. 📊 Overview
      - Multinomial Naive Bayes                    2. 💬 Single Comment Detection
                                                   3. 📁 Batch Comment Detection
                                                   4. 📈 Dataset Analytics
                                                   5. 🎯 Model Performance
                                                   6. ℹ️ About the Project
```
- **Local & Offline:** 100% functional on local Python runtime without paid API keys.
- **Cached Resource Loading:** Joblib models and NLTK corpora cached via `@st.cache_resource` for instant page responsiveness.

> **Speaker Notes:**  
> "Our Streamlit dashboard provides a production-grade user experience with dynamic model switching, animated progress bars, and zero external API dependencies."

---

### SLIDE 9: Dashboard Modules & Live Capabilities
- **💬 Single Comment Detection:**
  - Real-time text analysis with instant test presets.
  - Color-coded decision badges (🚨 **TOXIC** vs ✅ **NON-TOXIC**).
  - Linguistic Diagnostics expander showing active vocabulary tokens vs OOV words.
- **📁 Batch CSV Moderation:**
  - Upload arbitrary CSV files with auto-detected text columns.
  - Missing and blank comment handling.
  - Batch summary charts and one-click enriched CSV download.
  - Downloadable sample CSV for testing.
- **📈 Dataset Analytics:**
  - Ground-truth vs predicted distribution charts.
  - Toxicity subcategory breakdown and word length distributions.
- **🎯 Model Performance Hub:**
  - Confusion matrix heatmaps and classification reports loaded from verified JSON.

> **Speaker Notes:**  
> "The dashboard accommodates both individual users and platform administrators needing to batch moderate hundreds of comments at once."

---

### SLIDE 10: Live Demonstration Workflow & Conclusion
- **Demonstration Steps for Evaluators:**
  1. **Overview:** Review ground-truth metrics (200 records, 107 toxic, 93 non-toxic).
  2. **Single Detection:** Test *"You are an absolute idiot"* $\to$ Verify **TOXIC** classification and 59.1% probability.
  3. **Diagnostics:** Inspect tokenization and vocabulary matches.
  4. **Batch Processing:** Download sample CSV, upload, classify, and export enriched results.
  5. **Performance Benchmarking:** Review confusion matrices and comparative metrics.
- **Key Takeaway:** Successfully delivered a fully documented, rigorously tested, and interactive NLP classification system.

> **Speaker Notes:**  
> "To conclude, ToxicGuard demonstrates how classical NLP and statistical machine learning can be engineered into an end-to-end, trustworthy, and interactive moderation system. I invite the evaluators for questions and live demonstration."
