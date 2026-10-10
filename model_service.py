"""
Model loading, input validation, and prediction logic for the capstone demo.

This is kept separate from app.py on purpose: app.py only has to worry about
the Streamlit UI, and this module can be unit tested without Streamlit
running at all.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_VERSION = "rf_tuned_v1_capstone"
MODEL_PATH = PROJECT_ROOT / "models" / f"{MODEL_VERSION}.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / f"{MODEL_VERSION}_metadata.json"
LOG_PATH = PROJECT_ROOT / "logs" / "predictions.log"

VALID_COLORS = {"yellow", "green"}
VALID_BOROUGHS = {
    "Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR",
}


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is attempted before the model has loaded."""


class InvalidInputError(ValueError):
    """Raised when user-supplied input fails validation."""


class ModelService:
    """Loads the saved pipeline once and serves predictions on demand."""

    def __init__(self, model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
        self._model_path = model_path
        self._metadata_path = metadata_path
        self._pipeline = None
        self._metadata = None

    def load(self) -> None:
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self._model_path}. "
                "Run scripts/train_and_save_model.py first."
            )
        self._pipeline = joblib.load(self._model_path)
        if self._metadata_path.exists():
            with open(self._metadata_path) as f:
                self._metadata = json.load(f)
        else:
            self._metadata = {}

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def version(self) -> str:
        return self._metadata.get("model_version", MODEL_VERSION) if self._metadata else MODEL_VERSION

    def predict(self, record: dict) -> dict:
        if not self.is_loaded:
            raise ModelNotLoadedError("Model has not been loaded yet.")

        validate_record(record)

        import pandas as pd
        row = pd.DataFrame([record])
        prediction = self._pipeline.predict(row)[0]

        probability = None
        if hasattr(self._pipeline, "predict_proba"):
            proba = self._pipeline.predict_proba(row)[0]
            classes = list(self._pipeline.classes_)
            probability = float(proba[classes.index(prediction)])

        return {
            "prediction": prediction,
            "probability": probability,
            "model_version": self.version,
        }


def validate_record(record: dict) -> None:
    """Raise InvalidInputError with a clear message on any bad field."""
    required_fields = [
        "passengers", "distance", "fare", "tolls",
        "color", "pickup_borough", "dropoff_borough",
    ]
    for field in required_fields:
        if field not in record or record[field] in (None, ""):
            raise InvalidInputError(f"'{field}' is required.")

    passengers = record["passengers"]
    if not isinstance(passengers, int) or isinstance(passengers, bool):
        raise InvalidInputError("'passengers' must be a whole number.")
    if not (1 <= passengers <= 6):
        raise InvalidInputError("'passengers' must be between 1 and 6.")

    for field in ("distance", "fare", "tolls"):
        value = record[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise InvalidInputError(f"'{field}' must be a number.")
        if value < 0:
            raise InvalidInputError(f"'{field}' cannot be negative.")

    if record["color"] not in VALID_COLORS:
        raise InvalidInputError(f"'color' must be one of {sorted(VALID_COLORS)}.")

    for field in ("pickup_borough", "dropoff_borough"):
        if record[field] not in VALID_BOROUGHS:
            raise InvalidInputError(f"'{field}' must be one of {sorted(VALID_BOROUGHS)}.")


def log_prediction(record: dict, result: dict) -> None:
    """Append one structured JSON line per prediction for basic monitoring."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": record,
        "prediction": result["prediction"],
        "probability": result["probability"],
        "model_version": result["model_version"],
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
