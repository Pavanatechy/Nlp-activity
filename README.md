# ToxicGuard — AI-Powered Toxic Comment Detection Dashboard

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.66%2B-FF4B4B.svg)
![NLP](https://img.shields.io/badge/NLP-NLTK%20%7C%20Scikit--Learn-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed%20(Dashboard%20Phase)-brightgreen.svg)
![License](https://img.shields.io/badge/License-Academic%20MCA-purple.svg)

An academic Natural Language Processing (NLP) and Supervised Machine Learning web application designed to automatically detect, classify, and mitigate toxic, abusive, and hostile language in user-generated online comments.

Developed as part of the Master of Computer Applications (MCA) Mini-Project curriculum (**Activity 1, Activity 2, and Interactive Web Dashboard Deployment**).

---

## 📌 Project Overview & Problem Statement

Online harassment, cyberbullying, profanity, and toxic commentary undermine collaborative discourse on digital communities, talk pages, and forums. With millions of comments posted daily, manual human moderation is slow, unscalable, and psychologically damaging to moderators.

**ToxicGuard** provides an end-to-end Machine Learning and Natural Language Processing solution:
1. **Linguistic Preprocessing & Normalization:** Cleans and normalizes unstructured comments, removes noise, and standardizes vocabulary via lemmatization.
2. **TF-IDF Feature Engineering:** Extracts unigram and bigram statistics with sublinear frequency scaling, strictly fitted on training data to prevent data leakage.
3. **Dual Baseline Classification:** Benchmarks **Logistic Regression (Balanced)** against **Multinomial Naive Bayes (Laplace Smoothed)**.
4. **Interactive Streamlit Web Dashboard (`app.py`):** Real-time single comment inference, batch CSV moderation, dataset analytics, and model performance visualization.

---

## ✨ Key Features of the Dashboard

- **📊 Comprehensive Overview:**
  - Real-time display of ground-truth dataset statistics (200 total records: 107 toxic, 93 non-toxic).
  - Clear separation between ground-truth dataset metrics and active session telemetry.
  - Active model indicator, vocabulary size (764 n-grams), and test accuracy benchmark.
- **💬 Real-Time Single Comment Detection:**
  - Instant text evaluation with pre-configured preset buttons (Supportive, Direct Insult, Hostile Threat, OOV).
  - Prominent visual indicators (🚨 **TOXIC DETECTED** in crimson vs ✅ **NON-TOXIC / SAFE** in emerald).
  - Accurately labeled **model-estimated probabilities** ($P(\text{Toxic})$ vs $P(\text{Non-Toxic})$).
  - **Linguistic Transparency Diagnostics:** Shows cleaned tokens, recognized vocabulary words, and educational Out-of-Vocabulary (OOV) alerts.
- **📁 Bulk Batch Moderation via CSV Upload:**
  - Upload arbitrary CSV files with automatic text column detection (`comment_text`, `comment`, `text`, etc.).
  - Handles missing (`NaN`) or blank entries gracefully without crashing.
  - Generates batch summary metrics, toxicity distribution bar charts, and filterable data tables.
  - One-click export of enriched predictions to a downloadable CSV file.
  - Includes a downloadable sample CSV for immediate examiner testing.
- **📈 Ground-Truth Dataset Analytics:**
  - Class distribution bar charts (53.5% Toxic vs 46.5% Non-Toxic).
  - Human-annotated toxicity subcategories breakdown (*Obscene/Profanity*, *Insult*, *Threat*, *Identity Hate*, *Harassment*).
  - Comment length and word count distributions.
  - Interactive searchable dataset explorer.
- **🎯 Verified Model Performance & Diagnostics:**
  - Evaluated on the held-out test split ($N=40$, 80:20 stratified partition).
  - Exact verified metrics: Accuracy, Toxic Precision, Toxic Recall, Toxic F1, Macro F1, TP, TN, FP, FN.
  - Side-by-side confusion matrix heatmaps (Logistic Regression vs Naive Bayes).
  - Full classification reports and academic justification for selecting Logistic Regression.
- **ℹ️ Educational & Theoretical Documentation:**
  - Mathematical derivations of TF-IDF and Logistic Regression sigmoid formulas.
  - Detailed explanation of OOV intercept bias and system limitations.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Language** | Python 3.10+ (tested on Python 3.13) |
| **Web Dashboard** | Streamlit 1.66+, Altair |
| **NLP & Text Processing** | NLTK (Tokenization, Stopwords, WordNet Lemmatizer) |
| **Machine Learning** | Scikit-Learn (TF-IDF Vectorizer, Logistic Regression, MultinomialNB, Metrics) |
| **Data Manipulation** | Pandas, NumPy |
| **Data Visualization** | Matplotlib, Seaborn |
| **Model Persistence** | Joblib (`.pkl` serialization) |
| **Execution Environment** | Local Python runtime (100% offline, zero paid API keys) |

---

## 📂 Project Structure

```
Toxic-Comment-Detection/
│
├── Activity_1/
│   └── NLP.ipynb                                # Original Activity 1 annotation notebook
│
├── Activity_2/
│   └── Activity_2_Toxic_Comment_Detection.ipynb # Executed Activity 2 Notebook (All 15 sections)
│
├── src/
│   ├── train_model.py                           # Training pipeline for LR and MNB
│   ├── evaluate_model.py                        # Metric evaluation & JSON export
│   └── predict.py                               # CLI inference script
│
├── app.py                                       # Main Streamlit Web Dashboard
│
├── models/
│   ├── toxic_comment_model.pkl                  # Primary model (Logistic Regression, Joblib)
│   ├── tfidf_vectorizer.pkl                     # Fitted TF-IDF Vectorizer (Joblib)
│   └── naive_bayes_model.pkl                    # Baseline model (Multinomial Naive Bayes, Joblib)
│
├── results/
│   ├── model_evaluation_metrics.json            # Reusable JSON export of evaluation metrics
│   ├── class_distribution.png                   # Ground-truth class distribution chart
│   ├── confusion_matrices_combined.png          # Combined confusion matrix comparison
│   ├── confusion_matrix_lr.png                  # Logistic Regression confusion matrix
│   ├── confusion_matrix_nb.png                  # Naive Bayes confusion matrix
│   └── model_comparison.png                     # Side-by-side performance bar chart
│
├── toxic_comments_200_annotated.csv             # Curated human-annotated dataset (N=200)
├── train.csv                                    # Kaggle source benchmark corpus (159,571 rows)
├── requirements.txt                             # Python dependencies
├── README.md                                    # Comprehensive project documentation
├── Project_Report.md                            # Complete 9-section university project report
├── Presentation_Slides.md                       # Defense presentation slides
└── Viva_Questions.md                            # 15 technical viva Q&As + live demo script
```

---

## 🚀 Installation & Setup

### 1. Clone or Open the Workspace
```bash
cd "c:\Users\ADMIN\Downloads\NLP activity"
```

### 2. Set Up a Python Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

---

## 💻 How to Run the Project

### A. Launch the ToxicGuard Web Dashboard
Start the local Streamlit application:
```bash
streamlit run app.py
```
Open your browser at: **`http://localhost:8501`**

### B. Retrain the Models (Optional)
If you wish to retrain the classifiers from scratch and update saved artifacts:
```bash
python src/train_model.py
```
This will:
1. Load `toxic_comments_200_annotated.csv`.
2. Apply NLP text normalization and lemmatization.
3. Perform a stratified 80:20 split (160 train, 40 test).
4. Fit the TF-IDF vectorizer strictly on training data.
5. Train Logistic Regression and Multinomial Naive Bayes.
6. Serialize models to `models/`.

### C. Evaluate the Models and Export Metrics
Recompute the evaluation metrics on the held-out test split and update visualizations:
```bash
python src/evaluate_model.py
```
Generates `results/model_evaluation_metrics.json` and confusion matrix plots in `results/`.

### D. Run Command-Line Inference
Test arbitrary sentences directly from the terminal:
```bash
python src/predict.py --comment "You are an absolute idiot."
python src/predict.py --comment "Thank you for reviewing my edits."
```

---

## 📱 How to Use the Dashboard

### 1. Real-Time Single Comment Detection
1. Select **💬 Single Comment Detection** in the sidebar.
2. Click one of the quick preset buttons (*"Civil & Supportive"*, *"Direct Insult"*, *"Hostile Threat"*, *"Out-of-Vocabulary"*) or type your own comment.
3. Click **🔍 Analyze Comment**.
4. View the classification badge:
   - 🚨 **TOXIC**: Highlighted in red with estimated probability.
   - ✅ **NON-TOXIC**: Highlighted in green with estimated safety probability.
5. Expand the **Linguistic Diagnostics** dropdown to view tokenized words, recognized vocabulary n-grams, and OOV analysis.

### 2. Bulk Batch Comment Detection
1. Select **📁 Batch Comment Detection** in the sidebar.
2. (Optional) Click **📄 Download Sample Comments CSV** to obtain a ready-to-use test file.
3. Upload your CSV file via the file uploader.
4. Select the column containing text (automatically detected).
5. Click **🚀 Classify Batch Comments**.
6. View batch statistics (Total rows, Toxic %, Safe %, Skipped blanks).
7. Filter records by class or download the enriched file via **💾 Download Classified Results as CSV**.

### 3. Comparing Model Performance
1. Select **🎯 Model Performance** in the sidebar.
2. Inspect the side-by-side metric comparison table between **Logistic Regression** and **Multinomial Naive Bayes**.
3. View confusion matrix heatmaps and classification reports.
4. Review the **Selected Model Justification** card explaining the precision-recall trade-off.

---

## 📈 Measured Experimental Results

Evaluated on the independent test split ($N=40$ comments, stratified 80:20 split):

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

### Why Logistic Regression is the Selected Model
In automated moderation, **False Positives (Type I errors)** carry severe consequences: they silence legitimate user participation and damage community trust.
- **Naive Bayes generated 10 false positives** out of 19 non-toxic comments (a false alarm rate over **52%**).
- **Logistic Regression reduced false positives to 7**, boosting toxic precision from **64.29% to 70.83%** and overall F1-score from **73.47% to 75.56%**.
- Logistic Regression provides the optimal balance between high precision and reliable recall.

---

## ⚠️ Known Limitations & Academic Disclaimers

1. **Annotated Sample Size:** The curated experimental corpus contains 200 records ($N=160$ train, $N=40$ test). In a 40-instance test set, each individual sample carries a 2.5% impact on accuracy.
2. **Out-of-Vocabulary (OOV) Invariance:** TF-IDF vectorizers are restricted to terms observed in training. When an unobserved word (e.g., *'useless'*) is evaluated, its feature vector is all zeros ($\vec{x} = \mathbf{0}$) and the classifier defaults to its intercept bias ($P \approx 52.6\%$).
3. **Absence of Sarcasm and Syntactic Structure:** Linear bag-of-words classifiers evaluate co-occurrence frequencies without syntactic parse trees or bidirectional attention. Sarcastic praise cannot be distinguished from genuine praise.
4. **Binary Simplification:** Collapses multi-dimensional toxicity (threats, profanity, identity hate) into a single binary threshold.

---

## 🔮 Future Enhancements

- **Scale to Full Benchmark:** Train on 50,000+ observations from the full Kaggle `train.csv` benchmark.
- **Transformer Architectures:** Implement fine-tuned `bert-base-uncased` with HuggingFace Transformers to eliminate OOV issues and capture deep contextual nuance.
- **Multi-Label Classification:** Concurrently predict probabilities across all 6 toxicity dimensions (`severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`).

---

## 📜 Academic Integrity & Citation

This project is developed for educational purposes as part of the MCA degree curriculum.
- **Corpus Reference:** Wikipedia Toxic Comment Classification Challenge (Kaggle).
- **Frameworks:** Python, Scikit-Learn, Streamlit, NLTK, Pandas, NumPy, Matplotlib, Seaborn, Joblib.
