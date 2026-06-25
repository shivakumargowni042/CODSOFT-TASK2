"""
MovieMind — Training Script (cleaned)
Run: python train.py
Trains a CalibratedLinearSVC on the dataset in /dataset/movies_dataset.csv
and saves model + label_encoder to /artifacts/.
"""

import os
import re
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, classification_report

APP_DIR   = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(APP_DIR, "artifacts")
DATA_PATH = os.path.join(APP_DIR, "dataset", "movies_dataset.csv")

os.makedirs(ARTIFACTS, exist_ok=True)

# ── Preprocessing (must match app.py exactly) ─────────────────────────────────
STOPWORDS = set("""
a an the and or but in on at to of for is are was were be been being
have has had do does did will would shall should may might must can could
it its this that these those i me my we our you your he his she her they their them
with by from up about into over after what which who when where how all each every no not
""".split())

TAGLINES = [
    "the truth changes everything",
    "one wrong move means death",
    "the consequences are irreversible",
    "loyalties are tested",
    "nothing is what it seems",
    "the real enemy is closer than anyone thinks",
    "time is running out",
    "secrets have a way of surfacing",
    "every choice has a cost",
    "there is no going back",
]

def strip_taglines(text: str) -> str:
    text = text.lower()
    for tag in TAGLINES:
        text = text.replace(tag, "")
    return text.strip()

def preprocess(text: str) -> str:
    text = strip_taglines(text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["plot", "genre"])
    df["plot"]  = df["plot"].astype(str)
    df["genre"] = df["genre"].astype(str).str.strip()
    df = df[df["plot"].str.len() > 10].copy()

    # Remove exact duplicate rows (same plot + same genre)
    before = len(df)
    df = df.drop_duplicates(subset=["plot", "genre"])
    print(f"Removed {before - len(df)} exact duplicate rows")

    # Remove duplicate plot texts (keep first) to prevent data leakage
    before = len(df)
    df = df.drop_duplicates(subset=["plot"], keep="first")
    print(f"Removed {before - len(df)} duplicate plot texts")

    genres = sorted(df["genre"].unique().tolist())
    label_to_id = {g: i for i, g in enumerate(genres)}
    df["label"] = df["genre"].map(label_to_id)

    X = [preprocess(t) for t in df["plot"]]
    y = df["label"].values

    print(f"\nClean dataset: {len(X)} samples, {len(genres)} genres")
    print("Genres:", genres)
    for g in genres:
        print(f"  {g}: {(y == label_to_id[g]).sum()}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(
                ngram_range=(1, 3), max_features=30000,
                sublinear_tf=True, min_df=2
            )),
            ("char", TfidfVectorizer(
                ngram_range=(3, 6), max_features=20000,
                sublinear_tf=True, analyzer="char", min_df=2
            )),
        ])),
        ("clf", CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced", max_iter=5000),
            cv=5, method="sigmoid"
        )),
    ])

    print("\nTraining...")
    pipe.fit(X_tr, y_tr)

    y_pred = pipe.predict(X_te)
    macro_f1 = f1_score(y_te, y_pred, average="macro")
    print(f"Test macro F1: {macro_f1:.4f}")
    print()
    print(classification_report(y_te, y_pred, target_names=genres))

    # Save
    joblib.dump(pipe, os.path.join(ARTIFACTS, "model.pkl"))
    joblib.dump(
        {"genres": genres, "label_to_id": label_to_id,
         "id_to_label": {i: g for g, i in label_to_id.items()}},
        os.path.join(ARTIFACTS, "label_encoder.pkl")
    )
    with open(os.path.join(ARTIFACTS, "evaluation_report.json"), "w") as f:
        report = classification_report(y_te, y_pred, target_names=genres, output_dict=True)
        json.dump({
            "macro_f1": float(macro_f1),
            "accuracy": float((y_pred == y_te).mean()),
            "genres": genres,
            "n_samples": len(X),
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "per_genre": {g: report[g]["f1-score"] for g in genres},
        }, f, indent=2)

    print(f"\nModel saved to {ARTIFACTS}/model.pkl")


if __name__ == "__main__":
    main()
