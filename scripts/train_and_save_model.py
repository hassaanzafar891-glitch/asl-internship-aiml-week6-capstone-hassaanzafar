"""
Train and save the capstone model.

This is the "best-performing model from the internship": a Random Forest
classifier predicting how a NYC taxi ride was paid for (credit card vs cash),
using the exact tuned hyperparameters that won the comparison in Week 4
(validated again in Week 5's API). This script re-trains that pipeline from
scratch on the full training split so the capstone has its own clean,
reproducible model artifact instead of just reusing an old file.

Run it from the project root with the virtual environment active:
    python scripts/train_and_save_model.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv"

TARGET_COLUMN = "payment"
NUMERIC_FEATURES = ["passengers", "distance", "fare", "tolls"]
CATEGORICAL_FEATURES = ["color", "pickup_borough", "dropoff_borough"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Winning hyperparameters from the Week 4 RandomizedSearchCV comparison.
BEST_PARAMS = {
    "n_estimators": 249,
    "max_depth": 11,
    "min_samples_split": 4,
    "class_weight": None,
    "random_state": 42,
}

MODEL_VERSION = "rf_tuned_v1_capstone"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / f"{MODEL_VERSION}.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / f"{MODEL_VERSION}_metadata.json"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL)
    df = df.dropna(subset=[TARGET_COLUMN])
    return df


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    model = RandomForestClassifier(**BEST_PARAMS)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main():
    print("Loading data...")
    df = load_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training the tuned Random Forest pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    print("Evaluating on the held-out test split...")
    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, pos_label="credit card"), 4),
        "recall": round(recall_score(y_test, y_pred, pos_label="credit card"), 4),
        "f1": round(f1_score(y_test, y_pred, pos_label="credit card"), 4),
    }
    print("Test metrics:", metrics)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "RandomForestClassifier",
        "hyperparameters": BEST_PARAMS,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "target_classes": sorted(y.unique().tolist()),
        "test_metrics": metrics,
        "data_source": DATA_URL,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {METADATA_PATH}")


if __name__ == "__main__":
    main()
