"""
Resume classifier training script.

Usage:
    python model/train_model.py --data data/ --csv resume/resume.csv
    python model/train_model.py --csv resume/resume.csv
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.preprocess import clean_text
from utils.text_extractor import load_csv_dataset, load_pdf_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


MODEL_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / "resume_model.pkl"
META_PATH = MODEL_DIR / "model_meta.json"


def build_candidates() -> dict[str, Pipeline]:
    tfidf = dict(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=15000,
        sublinear_tf=True,
        min_df=2,
    )
    return {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf)),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")),
        ]),
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf)),
            ("clf", MultinomialNB(alpha=0.1)),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf)),
            ("clf", LinearSVC(C=1.0, max_iter=2000)),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf)),
            ("clf", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)),
        ]),
    }


def load_data(data_dir: str = None, csv_path: str = None) -> pd.DataFrame:
    frames = []

    if data_dir and Path(data_dir).exists():
        pdf_df = load_pdf_dataset(data_dir)
        if not pdf_df.empty:
            logger.info(f"PDF dataset: {len(pdf_df)} records")
            frames.append(pdf_df)

    if csv_path and Path(csv_path).exists():
        csv_df = load_csv_dataset(csv_path)
        if not csv_df.empty:
            logger.info(f"CSV dataset: {len(csv_df)} records")
            frames.append(csv_df)

    if not frames:
        raise ValueError("No data found. Provide --data or --csv with valid paths.")

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["text", "category"])
    df = df[df["text"].str.strip().astype(bool)]
    logger.info(f"Total samples after merge: {len(df)}")
    return df


def evaluate_model(pipeline, X_test, y_test, label_encoder) -> dict:
    preds = pipeline.predict(X_test)
    labels = label_encoder.classes_.tolist()
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="weighted", zero_division=0),
        "recall": recall_score(y_test, preds, average="weighted", zero_division=0),
        "f1": f1_score(y_test, preds, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(
            y_test, preds, target_names=labels, output_dict=True, zero_division=0
        ),
    }


def train(data_dir: str = None, csv_path: str = None, output_dir: str = None):
    df = load_data(data_dir, csv_path)

    logger.info("Preprocessing text...")
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 20]

    le = LabelEncoder()
    y = le.fit_transform(df["category"])
    X = df["clean_text"].values

    min_class = pd.Series(y).value_counts().min()
    if min_class < 2:
        logger.warning("Some classes have < 2 samples — stratified split may fail.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if min_class >= 2 else None
    )

    candidates = build_candidates()
    results = {}
    trained_pipelines = {}

    logger.info("Training and evaluating models...")
    for name, pipeline in candidates.items():
        logger.info(f"  → {name}")
        try:
            pipeline.fit(X_train, y_train)
            metrics = evaluate_model(pipeline, X_test, y_test, le)
            results[name] = metrics
            trained_pipelines[name] = pipeline
            logger.info(f"     Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")
        except Exception as e:
            logger.error(f"     Failed: {e}")

    if not results:
        raise RuntimeError("All models failed to train.")

    best_name = max(results, key=lambda k: results[k]["f1"])
    logger.info(f"\nBest model: {best_name} (F1={results[best_name]['f1']:.4f})")

    out_dir = Path(output_dir) if output_dir else MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "resume_model.pkl"
    meta_path = out_dir / "model_meta.json"

    bundle = {
        "pipeline": trained_pipelines[best_name],
        "label_encoder": le,
        "best_model_name": best_name,
    }
    joblib.dump(bundle, model_path)
    logger.info(f"Model saved → {model_path}")

    meta = {
        "best_model": best_name,
        "classes": le.classes_.tolist(),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "results": {
            name: {k: v for k, v in m.items() if k != "classification_report"}
            for name, m in results.items()
        },
        "classification_report": results[best_name]["classification_report"],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata saved → {meta_path}")

    return bundle, meta


def load_model(model_dir: str = None):
    path = Path(model_dir) / "resume_model.pkl" if model_dir else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run train_model.py first.")
    return joblib.load(path)


def load_meta(model_dir: str = None) -> dict:
    path = Path(model_dir) / "model_meta.json" if model_dir else META_PATH
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Resume Screening model")
    parser.add_argument("--data", type=str, default="data/", help="Directory with category subfolders of PDFs")
    parser.add_argument("--csv", type=str, default="resume/resume.csv", help="Path to resume CSV file")
    parser.add_argument("--output", type=str, default=None, help="Output directory for model files")
    args = parser.parse_args()

    train(data_dir=args.data, csv_path=args.csv, output_dir=args.output)
