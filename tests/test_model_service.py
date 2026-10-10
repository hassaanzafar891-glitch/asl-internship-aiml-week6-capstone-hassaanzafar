"""
Tests for the serving logic in model_service.py.

These test validation and prediction behavior directly, without going
through the Streamlit UI, since that's where the actual logic a reviewer
would want tested lives.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_service import InvalidInputError, ModelService, validate_record

VALID_RECORD = {
    "passengers": 1,
    "distance": 2.5,
    "fare": 12.0,
    "tolls": 0.0,
    "color": "yellow",
    "pickup_borough": "Manhattan",
    "dropoff_borough": "Brooklyn",
}


@pytest.fixture(scope="module")
def service():
    svc = ModelService()
    svc.load()
    return svc


def test_model_loads_successfully(service):
    assert service.is_loaded
    assert service.version


def test_valid_record_passes_validation():
    validate_record(VALID_RECORD)  # should not raise


def test_missing_field_raises():
    bad_record = dict(VALID_RECORD)
    del bad_record["distance"]
    with pytest.raises(InvalidInputError, match="distance"):
        validate_record(bad_record)


def test_wrong_type_raises():
    bad_record = dict(VALID_RECORD)
    bad_record["fare"] = "twelve dollars"
    with pytest.raises(InvalidInputError, match="fare"):
        validate_record(bad_record)


def test_negative_value_raises():
    bad_record = dict(VALID_RECORD)
    bad_record["tolls"] = -5.0
    with pytest.raises(InvalidInputError, match="tolls"):
        validate_record(bad_record)


def test_too_many_passengers_raises():
    bad_record = dict(VALID_RECORD)
    bad_record["passengers"] = 10
    with pytest.raises(InvalidInputError, match="passengers"):
        validate_record(bad_record)


def test_invalid_color_raises():
    bad_record = dict(VALID_RECORD)
    bad_record["color"] = "purple"
    with pytest.raises(InvalidInputError, match="color"):
        validate_record(bad_record)


def test_invalid_borough_raises():
    bad_record = dict(VALID_RECORD)
    bad_record["pickup_borough"] = "Atlantis"
    with pytest.raises(InvalidInputError, match="pickup_borough"):
        validate_record(bad_record)


def test_predict_returns_expected_shape(service):
    result = service.predict(VALID_RECORD)
    assert result["prediction"] in {"credit card", "cash"}
    assert 0.0 <= result["probability"] <= 1.0
    assert result["model_version"] == service.version


def test_predict_rejects_invalid_input(service):
    bad_record = dict(VALID_RECORD)
    bad_record["passengers"] = -1
    with pytest.raises(InvalidInputError):
        service.predict(bad_record)
