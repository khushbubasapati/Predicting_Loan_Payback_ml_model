import pytest
import os
import pickle
import json


def test_model_file_exists():
    """Test if model file exists"""
    assert os.path.exists("model/model.pkl")


def test_model_metadata_exists():
    """Test if metadata exists"""
    assert os.path.exists("model/model_metadata.json")


def test_model_can_load():
    """Test model loading"""
    with open("model/model.pkl", 'rb') as f:
        model = pickle.load(f)
    assert model is not None


def test_model_metadata_content():
    """Test metadata has required fields"""
    with open("model/model_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    assert "optuna_cv_auc" in metadata or "val_auc" in metadata