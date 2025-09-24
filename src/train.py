"""
Sentiment classifier training and inference.

This script builds and compares two models:
1. A simple baseline (TF-IDF + Logistic Regression).
2. A final, highly optimized version that combines TF-IDF features with
   sentiment intensity features from VADER. It uses GridSearchCV to
   systematically find the best possible hyperparameters for this advanced
   pipeline, aiming for the highest possible accuracy.

Usage:
    python src/train.py
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Configuration ---
TRAIN_PATH = os.path.join("data", "sentiment", "train.csv")
DEV_PATH = os.path.join("data", "sentiment", "dev.csv")
TEST_PATH = os.path.join("data", "sentiment", "test.csv")
OUTPUT_PATH = os.path.join("submissions", "sentiment_test_predictions.csv")

os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)


# --- Custom Feature Engineering ---
class VaderSentimentFeatures(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer that extracts VADER sentiment scores
    (positive, negative, neutral, compound) from a list of texts.
    """
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        features = np.array([list(self.analyzer.polarity_scores(text).values()) for text in texts])
        return features


# --- Data Loading and Preprocessing ---
def load_split(path: str) -> pd.DataFrame:
    """Loads a CSV dataset."""
    return pd.read_csv(path)


def simple_clean(text: str) -> str:
    """A lightweight text cleaning function."""
    return text.strip().lower() if isinstance(text, str) else ""


def normalize_label(value: any) -> int:
    """Converts various label formats into a standard binary format (0 or 1)."""
    if isinstance(value, str) and value.strip().lower() in {"1", "positive", "pos"}:
        return 1
    try:
        if int(value) > 0: return 1
    except (ValueError, TypeError): pass
    return 0


# --- Model Building ---
def build_baseline() -> Pipeline:
    """Builds a baseline model: TF-IDF + Logistic Regression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)),
    ])


def build_and_tune_advanced_model(X_train: list, y_train: list) -> GridSearchCV:
    """
    Builds and tunes a high-performing pipeline that combines TF-IDF and VADER
    features, feeding them into a LinearSVC classifier.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('tfidf', TfidfVectorizer(preprocessor=simple_clean)),
            ('vader', VaderSentimentFeatures())
        ])),
        ('clf', LinearSVC(random_state=42, max_iter=3000, dual=False))
    ])

    parameters = {
        'features__tfidf__ngram_range': [(1, 2), (1, 3)],
        'features__tfidf__sublinear_tf': [True, False],
        'clf__C': [0.5, 1, 5, 10],
        'clf__class_weight': ['balanced']
    }

    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, scoring='f1_macro')
    print("[Sentiment] Starting advanced hyperparameter search (TF-IDF + VADER)...")
    grid_search.fit(X_train, y_train)
    
    print(f"[Sentiment] Best parameters found: {grid_search.best_params_}")
    return grid_search


# --- Evaluation and Prediction ---
def evaluate(model, X: list, y: list, title: str) -> None:
    """Evaluates the model and prints a detailed performance report."""
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro")
    print(f"\n[{title}] Dev Set Performance:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1-Score: {f1:.4f}")
    print(classification_report(y, preds, digits=4, zero_division=0))


def main() -> None:
    """Main function to run the sentiment analysis pipeline."""
    print("[Sentiment] Loading data...")
    for p in [TRAIN_PATH, DEV_PATH, TEST_PATH]:
        if not os.path.exists(p):
            sys.exit(f"Error: Required data file not found at '{p}'")

    train_df, dev_df, test_df = map(load_split, [TRAIN_PATH, DEV_PATH, TEST_PATH])

    train_df["label"] = train_df["label"].apply(normalize_label)
    dev_df["label"] = dev_df["label"].apply(normalize_label)

    X_train, y_train = train_df["text"].fillna("").tolist(), train_df["label"].tolist()
    X_dev, y_dev = dev_df["text"].fillna("").tolist(), dev_df["label"].tolist()
    X_test = test_df["text"].fillna("").tolist()

    baseline_model = build_baseline()
    baseline_model.fit(X_train, y_train)
    evaluate(baseline_model, X_dev, y_dev, "Baseline")

    best_model = build_and_tune_advanced_model(X_train, y_train)
    evaluate(best_model, X_dev, y_dev, "Advanced (TF-IDF + VADER + Tuned SVC)")

    print("\n[Sentiment] Performing error analysis on dev set (up to 5 examples)...")
    dev_preds = best_model.predict(X_dev)
    mistakes = [
        (i, X_dev[i], y_dev[i], dev_preds[i])
        for i, (gold, pred) in enumerate(zip(y_dev, dev_preds))
        if gold != pred
    ]
    for i, text, gold, pred in mistakes[:5]:
        print(f"- Index={i} | Gold={gold} | Pred={pred} | Text='{text[:200]}'")

    print("\n[Sentiment] Predicting on the test set and saving results...")
    test_predictions = best_model.predict(X_test)
    output_df = pd.DataFrame({"text": test_df["text"], "label": test_predictions})
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"[Sentiment] Saved predictions to '{OUTPUT_PATH}'")


if __name__ == "__main__":
    main()

