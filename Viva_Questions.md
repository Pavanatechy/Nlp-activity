# ToxicGuard: AI-Powered Toxic Comment Detection
## Comprehensive Viva Questions & Examination Defense Guide
**Master of Computer Applications (MCA) — Academic Mini-Project**  
**Activity 2 & Dashboard Phase**

---

### PART 1: CORE NLP & MACHINE LEARNING ARCHITECTURE

#### Q1: What is the problem statement and objective of this project?
> **Answer:**  
> The project addresses the proliferation of toxic, aggressive, and abusive comments on online community platforms (such as Wikipedia talk pages). Because manual moderation cannot scale to millions of messages and causes severe psychological fatigue for human moderators, our objective is to build an automated, interpretable, and lightweight NLP classification system that flags comments as **Toxic (1)** or **Non-Toxic (0)**, complete with confidence scores and a full-featured Streamlit web dashboard.

---

#### Q2: What dataset was used, and how was it prepared?
> **Answer:**  
> We utilized two datasets:
> 1. **Kaggle Benchmark Corpus (`train.csv`):** 159,571 Wikipedia talk page comments across six multi-label targets (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`). This dataset has severe class imbalance (only 9.58% toxic comments).
> 2. **Curated Project Dataset (`toxic_comments_200_annotated.csv`):** A curated 200-sample corpus balanced between originally toxic ($N=100$) and non-toxic ($N=100$) samples, refined through human manual annotation in Activity 1 into:
>    - **Toxic (1):** 107 instances (53.5%)
>    - **Non-Toxic (0):** 93 instances (46.5%)
>    - Subcategories annotated: *Obscene/Profanity* (45), *Insult* (24), *Other* (22), *Identity Hate* (7), *Threat* (5), *Harassment* (4).

---

#### Q3: What is Data Leakage, and how did you prevent it?
> **Answer:**  
> **Data Leakage** occurs when information from outside the training partition is inadvertently used to train the model or extract features—causing overly optimistic performance estimates during training that fail in production.
> 
> In this project, **we strictly prevented data leakage** by performing the stratified 80:20 train-test partition *before* vocabulary extraction or TF-IDF fitting:
> - **Training partition ($80\%$):** 160 comments (86 Toxic, 74 Non-Toxic).
> - **Testing partition ($20\%$):** 40 comments (21 Toxic, 19 Non-Toxic).
> 
> The `TfidfVectorizer` was fitted strictly on `X_train` (`fit_transform`). The test set `X_test` was only mapped using `transform`, ensuring zero leakage of test frequencies or vocabulary terms.

---

#### Q4: Explain the NLP preprocessing pipeline in detail.
> **Answer:**  
> The preprocessing pipeline follows four sequential linguistic stages:
> 1. **Text Normalization:**
>    - Case conversion: Lowercasing all characters.
>    - Regex stripping: Eliminating HTML tags (`<.*?>`), web URLs (`http\S+|www\S+`), and newlines/tabs (`[\n\t\r]`).
>    - Whitespace standardization: Collapsing multiple spaces into single space.
> 2. **Tokenization:** NLTK `word_tokenize` breaks sentences into lexical tokens.
> 3. **Stop-word Removal:** NLTK English stop-words are eliminated, as well as single-character tokens (`len(w) > 1`), discarding non-discriminative syntax tokens (*'the'*, *'is'*, *'and'*).
> 4. **Morphological Lemmatization:** NLTK `WordNetLemmatizer` reduces inflectional suffixes to their root dictionary form (*'running'* $\to$ *'run'*, *'idiots'* $\to$ *'idiot'*), reducing vector space sparsity.

---

#### Q5: How does TF-IDF feature extraction work mathematically?
> **Answer:**  
> TF-IDF evaluates how important a word is to a document relative to the whole corpus:
> 
> $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
> 
> - **Sublinear Term Frequency:** $\text{TF}(t, d) = 1 + \log(\text{count}(t, d))$ (sublinear scaling down-weights words that repeat excessively in a single comment).
> - **Smooth Inverse Document Frequency:**
>   $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + \text{DF}(t, D)}\right) + 1$$
>   penalizes words appearing in almost every document.
> - **N-Gram Range $(1, 2)$:** We extracted unigrams and bigrams (`ngram_range=(1, 2)`, `min_df=2`), allowing the model to capture critical phrases such as *"shut up"* or *"go away"* that carry toxic meaning when grouped.
> - **Learned Vocabulary:** 764 distinct n-gram features on the training set.

---

#### Q6: Which machine learning algorithms were trained and compared?
> **Answer:**  
> We trained and benchmarked two complementary classifiers:
> 1. **Logistic Regression (Selected Model):**
>    - Discriminative linear classification model optimizing log-loss.
>    - Employs `class_weight='balanced'` to inversely scale weights by class frequencies.
>    - Yields calibrated probabilities via the sigmoid link function $\sigma(z) = \frac{1}{1 + e^{-z}}$.
> 2. **Multinomial Naive Bayes (Baseline Model):**
>    - Generative probabilistic model based on Bayes' theorem with the conditional independence assumption.
>    - Applies **Laplace Smoothing** ($\alpha=1.0$) to prevent zero-probability errors on rare features.

---

### PART 2: MODEL PERFORMANCE & EVALUATION METRICS

#### Q7: What are the exact evaluation results on the held-out test split ($N=40$)?
> **Answer:**  
> Evaluated on the independent test split ($N=40$ comments; 21 Toxic, 19 Non-Toxic):
> 
> | Metric | Logistic Regression (Selected) | Multinomial Naive Bayes | Best Performer |
> | :--- | :---: | :---: | :---: |
> | **Accuracy** | **72.50%** | 67.50% | **Logistic Regression** (+5.00%) |
> | **Toxic Precision** | **70.83%** | 64.29% | **Logistic Regression** (+6.54%) |
> | **Toxic Recall** | 80.95% | **85.71%** | Multinomial Naive Bayes (+4.76%) |
> | **Toxic F1-Score** | **75.56%** | 73.47% | **Logistic Regression** (+2.09%) |
> | **Macro F1-Score** | **72.06%** | 65.77% | **Logistic Regression** (+6.29%) |
> | **True Positives (TP)** | 17 | 18 | MNB |
> | **True Negatives (TN)** | **12** | 9 | **Logistic Regression** |
> | **False Positives (FP)** | **7** | 10 | **Logistic Regression (Fewer errors)** |
> | **False Negatives (FN)** | 4 | **3** | MNB |

---

#### Q8: Why did you select Logistic Regression over Multinomial Naive Bayes?
> **Answer:**  
> In automated content moderation, **False Positives (Type I error)** carry severe consequences: they censor safe, non-toxic comments and frustrate harmless users.
> 
> - **Naive Bayes generated 10 False Positives** out of 19 non-toxic comments—a false alarm rate exceeding **52.6%**.
> - **Logistic Regression produced only 7 False Positives**, achieving **70.83% Precision** (vs. 64.29% for MNB) and a higher overall **F1-Score of 75.56%** (vs. 73.47%).
> - Logistic Regression provides superior precision without a significant sacrifice in recall, making it safer for real-world moderation.

---

#### Q9: How do you handle Out-of-Vocabulary (OOV) words? Why does a sentence like "You are useless" get a borderline score?
> **Answer:**  
> In bag-of-words and TF-IDF models, any token not observed in the training set receives a zero weight vector $\vec{x} = \mathbf{0}$.
> 
> When input vector $\mathbf{x} = \mathbf{0}$, the Logistic Regression prediction is determined purely by the intercept term $b$:
> 
> $$P(\text{Toxic}|\mathbf{0}) = \sigma(\mathbf{w}^T \mathbf{0} + b) = \frac{1}{1 + e^{-b}}$$
> 
> In our trained model, $b \approx 0.1035$, so $\sigma(0.1035) \approx 0.5258$ (52.6%).
> 
> This is a known, mathematically honest limitation of small-vocabulary TF-IDF models, and our dashboard explicitly detects and explains OOV occurrences through an educational diagnostic card.

---

### PART 3: STREAMLIT DASHBOARD & SYSTEM DEPLOYMENT

#### Q10: Describe the architecture and design of the ToxicGuard Dashboard (`app.py`).
> **Answer:**  
> The dashboard is developed in **Python using Streamlit** and organized into six modular sections:
> 1. **Sidebar:** Project metadata, dynamic navigation, model switcher (Logistic Regression vs Naive Bayes), and session telemetry.
> 2. **Overview:** Ground-truth dataset metrics (200 total, 107 toxic, 93 non-toxic), active model, vocabulary size (764), and session counters.
> 3. **Single Comment Detection:** Real-time text classification with preset examples, input validation, animated probability progress bars, and linguistic feature inspector (active vocabulary tokens vs OOV tokens).
> 4. **Batch Comment Detection:** Multi-comment CSV upload, auto-column detection, missing value handling, batch prediction, probability calculation, and filtered export of results as CSV.
> 5. **Dataset Analytics:** Ground-truth class distribution bar charts, subcategory breakdowns, comment length distributions, and dataset search engine.
> 6. **Model Performance:** Side-by-side benchmarking, classification reports, confusion matrix heatmaps, and false-positive vs false-negative tradeoff analysis.
> 7. **About the Project:** Comprehensive documentation of mathematical formulas, NLP pipelines, and academic limitations.

---

#### Q11: How is model loading optimized in Streamlit?
> **Answer:**  
> We use `@st.cache_resource` for loading the model and vectorizer artifacts (`toxic_comment_model.pkl`, `naive_bayes_model.pkl`, `tfidf_vectorizer.pkl`) and downloading NLTK tokenizers.
> 
> This ensures that heavy files are read from disk **only once** when the server initializes, rather than re-reading or re-training upon every user interaction. Dataframes and evaluation metrics use `@st.cache_data`.

---

#### Q12: How would you demonstrate the live application during the viva?
> **Answer:**  
> **Viva Demonstration Flow (3-Minute Script):**
> 1. **Overview (30s):** Start on `📊 Overview`. Highlight the verified ground-truth metrics (200 total comments: 107 toxic, 93 non-toxic) and active model (Logistic Regression). Point out that session statistics are clearly separated from dataset records.
> 2. **Single Comment Detection (60s):** Navigate to `💬 Single Comment Detection`.
>    - Click the preset *"Direct Insult"* $\to$ Click *"Analyze Comment"* $\to$ Show red **TOXIC** badge with 59.1% probability.
>    - Click the preset *"Civil & Supportive"* $\to$ Show green **NON-TOXIC** badge with 53.7% safety probability.
>    - Open the **Linguistic Diagnostics** expander $\to$ Show which tokens matched the TF-IDF vocabulary and explain how lemmatization standardized suffixes.
> 3. **Batch Moderation (45s):** Navigate to `📁 Batch Comment Detection`. Click *"Download Sample Comments CSV"*, upload it, and click *"Classify Batch Comments"*. Show the summary metrics, distribution chart, filter options, and download the classified CSV.
> 4. **Model Performance (45s):** Navigate to `🎯 Model Performance`. Show the exact test metrics (72.50% Acc, 75.56% F1), point to the confusion matrices, and explain why Logistic Regression was selected over Naive Bayes due to having 3 fewer false positives.

---

#### Q13: What are the primary limitations of the current implementation?
> **Answer:**  
> 1. **Small Annotated Sample Size:** Curated on 200 comments ($N=160$ train, $N=40$ test). Each test instance accounts for 2.5% of accuracy.
> 2. **Out-of-Vocabulary (OOV) Invariance:** Words not seen during training cannot contribute to TF-IDF weights.
> 3. **Absence of Sarcasm and Syntactic Context:** Linear bag-of-words classifiers evaluate co-occurrence frequencies without syntactic trees or self-attention.
> 4. **Binary Simplification:** Collapses severe threats, obscene profanity, and mild insults into a single binary threshold.

---

#### Q14: How would you enhance the project for future phases?
> **Answer:**  
> 1. **Scale to Full Benchmark:** Train on 50,000+ observations from the full Kaggle `train.csv` corpus.
> 2. **Transformer Architectures (BERT / RoBERTa):** Implement fine-tuned `bert-base-uncased` with HuggingFace Transformers. Subword WordPiece tokenization eliminates the OOV problem, and multi-head self-attention captures sarcastic and bidirectional contextual meaning.
> 3. **Multi-Label Classification:** Concurrently predict probabilities across all 6 toxicity dimensions (`severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`).
> 4. **Containerization & CI/CD:** Package the Streamlit app into a Docker container and deploy via GCP Cloud Run or AWS ECS.

---

#### Q15: How do you verify that your metrics are real and not fabricated?
> **Answer:**  
> All metrics displayed in the dashboard originate directly from `results/model_evaluation_metrics.json`, generated by running `src/evaluate_model.py` on the held-out test split ($N=40$). The test partition indices are strictly seeded (`random_state=42`) and stratified.
> 
> Any evaluator can re-run `python src/evaluate_model.py` or inspect `Activity_2/Activity_2_Toxic_Comment_Detection.ipynb` to reproduce the exact identical confusion matrices and classification reports down to the decimal point.
