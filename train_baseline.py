import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from dataset import load_review_data

def train_baseline(
    csv_path="data/reviews_imdb_5k.csv",
    model_path="models/baseline_lr.joblib"
):
    # Load dataset
    df = load_review_data(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")

    texts = df["text"]
    labels = df["label"]

    # Train/test split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    print(f"Train size: {len(train_texts)}")
    print(f"Test size:  {len(test_texts)}")

    # Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2)
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # Model
    clf = LogisticRegression(max_iter=3000, n_jobs=-1)
    clf.fit(X_train, train_labels)

    # Predictions
    preds = clf.predict(X_test)

    # Metrics
    acc = accuracy_score(test_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, preds, average="weighted"
    )

    print("\n=== BASELINE MODEL RESULTS (IMDB 5K) ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "model": clf}, model_path)
    print(f"\nModel saved to: {model_path}")

    return acc, precision, recall, f1


if __name__ == "__main__":
    train_baseline()
