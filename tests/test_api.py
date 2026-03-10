"""Tests for FastAPI application"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

from api.app import app

client = TestClient(app)


def test_read_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Loan Prediction API"


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OK"
    assert "version" in data
    assert "model_loaded" in data
    assert data["model_loaded"] is True


def test_predict_valid_data():
    """Test prediction with valid data"""
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
    
    # Debug output if fails
    if response.status_code != 200:
        print("\nError Response:")
        print(response.json())
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert isinstance(data, dict)
   
def test_predict_invalid_data():
    """Test prediction with invalid data"""
    payload = {
        "annual_income": -1000.0,  # Invalid: must be > 0
        "debt_to_income_ratio": 0.5,
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
    # Should return 422 validation error
    assert response.status_code == 422


def test_predict_out_of_range_debt_ratio():
    """Test with debt_to_income_ratio > 1"""
    payload = {
        "annual_income": 75000.0,
        "debt_to_income_ratio": 1.5,  # Invalid: must be 0-1
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


def test_predict_invalid_credit_score():
    """Test with credit score out of range"""
    payload = {
        "annual_income": 75000.0,
        "debt_to_income_ratio": 0.35,
        "credit_score": 200,  # Invalid: must be 300-850
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
    """Test prediction with missing required fields"""
    payload = {
        "annual_income": 75000.0
        # Missing all other required fields
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422