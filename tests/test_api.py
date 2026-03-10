import pytest
from fastapi.testclient import TestClient
import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

mock_predict = MagicMock()
mock_predict.model = MagicMock()
mock_predict.MODEL_VERSION = 'v1.0.0'
mock_predict.predict_output = MagicMock(return_value={
    "loan_paid_back_prediction": 1,
    "loan_paid_back_probability": 0.85
})

sys.modules['model.predict'] = mock_predict

from api.app import app
client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "OK"

def test_predict_valid_data():
    payload = {
        "annual_income": 75000.0,
        "debt_to_income_ratio": 0.355,
        "credit_score": 720,
        "loan_amount": 25000.0,
        "interest_rate": 8.5,
        "gender": "Male",
        "marital_status": "Married",
        "education_level": "Bachelor's",
        "employment_status": "Employed",
        "loan_purpose": "Debt consolidation",
        "grade_subgrade": "C3"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

def test_predict_invalid_debt_ratio():
    payload = {
        "annual_income": 75000.0,
        "debt_to_income_ratio": 1.5,
        "credit_score": 720,
        "loan_amount": 25000.0,
        "interest_rate": 8.5,
        "gender": "Male",
        "marital_status": "Married",
        "education_level": "Bachelor's",
        "employment_status": "Employed",
        "loan_purpose": "Debt consolidation",
        "grade_subgrade": "C3"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_missing_fields():
    payload = {"annual_income": 75000.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422