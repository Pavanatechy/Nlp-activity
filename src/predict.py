"""
predict.py
==========
Interactive inference script for Toxic Comment Detection.
Loads saved TF-IDF vectorizer and trained classification model,
preprocesses new unseen comments, and outputs predicted category & confidence.
"""

import os
import re
import argparse
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[\n\t\r]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_text(text, stop_words, lemmatizer):
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    filtered = [w for w in tokens if w not in stop_words and len(w) > 1]
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered]
    return " ".join(lemmatized)

class ToxicCommentPredictor:
    def __init__(self, model_path="models/toxic_comment_model.pkl", vectorizer_path="models/tfidf_vectorizer.pkl"):
        # Resolve path if run from subdirectory
        if not os.path.exists(model_path):
            alt_m = os.path.join("..", model_path)
            alt_v = os.path.join("..", vectorizer_path)
            if os.path.exists(alt_m) and os.path.exists(alt_v):
                model_path, vectorizer_path = alt_m, alt_v
            else:
                raise FileNotFoundError(f"Artifacts not found at {model_path} or {alt_m}")

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        
        # Ensure NLTK resources
        for r in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
            try:
                nltk.download(r, quiet=True)
            except Exception:
                pass
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def predict(self, comment_text):
        processed = preprocess_text(comment_text, self.stop_words, self.lemmatizer)
        tfidf_vec = self.vectorizer.transform([processed])
        
        pred_label_idx = self.model.predict(tfidf_vec)[0]
        
        # Check if model supports predict_proba
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(tfidf_vec)[0]
            confidence = probs[pred_label_idx]
            prob_toxic = probs[1]
        else:
            confidence = 1.0
            prob_toxic = 1.0 if pred_label_idx == 1 else 0.0

        label_str = "Toxic" if pred_label_idx == 1 else "Non-Toxic"
        
        # Check active vocabulary overlap
        feature_names = self.vectorizer.get_feature_names_out()
        tokens_in_vocab = [w for w in processed.split() if w in feature_names]

        return {
            "original_comment": comment_text,
            "processed_text": processed,
            "prediction": label_str,
            "confidence": confidence,
            "prob_toxic": prob_toxic,
            "prob_non_toxic": 1.0 - prob_toxic,
            "tokens_in_vocab": tokens_in_vocab
        }

def run_demo():
    predictor = ToxicCommentPredictor()
    
    sample_comments = [
        "Thank you for helping me.",
        "You are an idiot.",
        "I appreciate your hard work.",
        "You are useless."
    ]

    print("\n" + "="*70)
    print("               LIVE TOXIC COMMENT PREDICTION DEMONSTRATION         ")
    print("="*70)

    for i, comment in enumerate(sample_comments, 1):
        res = predictor.predict(comment)
        print(f"\n[Comment {i}]: \"{res['original_comment']}\"")
        print(f" -> Processed Tokens : {res['processed_text']}")
        print(f" -> Tokens in Vocab  : {res['tokens_in_vocab']}")
        print(f" -> Predicted Class  : {res['prediction'].upper()}")
        print(f" -> Confidence       : {res['confidence']*100:.2f}% (P(Toxic)={res['prob_toxic']:.4f})")
        
        # Note on contextual limitation
        if not res["tokens_in_vocab"]:
            print("    [NOTE]: All tokens are out-of-vocabulary (OOV). Prediction defaults to dataset prior.")
        elif res["prediction"] == "Toxic" and res["prob_toxic"] < 0.60:
            print("    [NOTE]: Marginal confidence score; prediction is close to decision boundary.")

    print("\n" + "="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict toxicity of comment")
    parser.add_argument("--comment", type=str, default=None, help="Comment string to evaluate")
    args = parser.parse_args()

    if args.comment:
        predictor = ToxicCommentPredictor()
        result = predictor.predict(args.comment)
        print(f"\nComment: \"{result['original_comment']}\"")
        print(f"Predicted Class: {result['prediction']}")
        print(f"Confidence: {result['confidence']*100:.2f}% (P(Toxic)={result['prob_toxic']:.4f})")
    else:
        run_demo()
