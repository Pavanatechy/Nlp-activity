"""
train_model.py
==============
Trains baseline NLP models (Logistic Regression and Multinomial Naive Bayes)
for Toxic Comment Detection using TF-IDF feature extraction.
Saves the trained model and TF-IDF vectorizer to disk using Joblib.
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import joblib

def ensure_nltk_resources():
    """Download required NLTK resources silently if missing."""
    resources = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
    for r in resources:
        try:
            nltk.download(r, quiet=True)
        except Exception:
            pass

def clean_text(text):
    """
    Apply text normalization as defined in Activity 1:
    - Lowercasing
    - Removing HTML tags
    - Removing URLs
    - Replacing newlines and tabs
    - Collapsing multiple whitespace
    """
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[\n\t\r]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_text(text, stop_words, lemmatizer):
    """
    Complete NLP text preprocessing pipeline:
    Cleaning -> Tokenization -> Stop-word filtering -> Lemmatization.
    """
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    filtered = [w for w in tokens if w not in stop_words and len(w) > 1]
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered]
    return " ".join(lemmatized)

def train(data_path="toxic_comments_200_annotated.csv", models_dir="models"):
    """
    Main training routine:
    1. Loads dataset.
    2. Preprocesses text comments.
    3. Stratified train-test split (80:20).
    4. Fits TF-IDF on training set only.
    5. Trains Logistic Regression & Multinomial Naive Bayes.
    6. Persists selected model & vectorizer.
    """
    ensure_nltk_resources()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    # Smart dataset path resolution
    if not os.path.exists(data_path):
        alt_path = os.path.join("..", data_path)
        if os.path.exists(alt_path):
            data_path = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at {data_path} or {alt_path}")

    print(f"[INFO] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[INFO] Dataset shape: {df.shape}")

    # Determine target column
    if "manual_label" in df.columns:
        print("[INFO] Target column: 'manual_label' (Activity 1 Annotated Dataset)")
        label_map = {"Toxic": 1, "Non-Toxic": 0}
        df["target"] = df["manual_label"].map(label_map)
    elif "toxic" in df.columns:
        print("[INFO] Target column: 'toxic'")
        df["target"] = df["toxic"].astype(int)
    else:
        raise ValueError("Neither 'manual_label' nor 'toxic' column found in dataset!")

    print("[INFO] Class distribution:")
    print(df["target"].value_counts())

    # Preprocess text
    print("[INFO] Running text preprocessing (cleaning, tokenization, lemmatization)...")
    df["processed_text"] = df["comment_text"].apply(lambda t: preprocess_text(t, stop_words, lemmatizer))

    # Stratified Train/Test Split (prevents data leakage)
    print("[INFO] Performing 80:20 stratified train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["processed_text"],
        df["target"],
        test_size=0.20,
        random_state=42,
        stratify=df["target"]
    )
    print(f"[INFO] Training set size: {len(X_train)} (Toxic: {sum(y_train==1)}, Non-Toxic: {sum(y_train==0)})")
    print(f"[INFO] Testing set size:  {len(X_test)} (Toxic: {sum(y_test==1)}, Non-Toxic: {sum(y_test==0)})")

    # Feature Extraction: TF-IDF fitted STRICTLY on X_train
    print("[INFO] Fitting TF-IDF Vectorizer on X_train only (preventing data leakage)...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=1000,
        min_df=2,
        sublinear_tf=True
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"[INFO] TF-IDF Matrix Shape (Train): {X_train_tfidf.shape}")
    print(f"[INFO] TF-IDF Matrix Shape (Test):  {X_test_tfidf.shape}")

    # Model 1: Logistic Regression
    print("[INFO] Training Model 1: Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, class_weight="balanced")
    lr_model.fit(X_train_tfidf, y_train)

    # Model 2: Multinomial Naive Bayes
    print("[INFO] Training Model 2: Multinomial Naive Bayes...")
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train_tfidf, y_train)

    # Create models directory
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "toxic_comment_model.pkl")
    vectorizer_save_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    nb_save_path = os.path.join(models_dir, "naive_bayes_model.pkl")

    # Save artifacts using Joblib
    # We save Logistic Regression as the primary selected model due to higher F1-score & precision
    joblib.dump(lr_model, model_save_path)
    joblib.dump(tfidf, vectorizer_save_path)
    joblib.dump(nb_model, nb_save_path)
    print(f"[SUCCESS] Saved primary model to: {model_save_path}")
    print(f"[SUCCESS] Saved TF-IDF vectorizer to: {vectorizer_save_path}")
    print(f"[SUCCESS] Saved Naive Bayes model to: {nb_save_path}")

    return {
        "lr_model": lr_model,
        "nb_model": nb_model,
        "tfidf": tfidf,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_test_tfidf": X_test_tfidf
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Toxic Comment Detection Models")
    parser.add_argument("--data", default="toxic_comments_200_annotated.csv", help="Path to input CSV dataset")
    parser.add_argument("--models_dir", default="models", help="Directory to save models")
    args = parser.parse_args()
    train(data_path=args.data, models_dir=args.models_dir)
