from fastapi import FastAPI
from fastapi.responses import JSONResponse
from schema.user_input import UserInput
from schema.prediction_response import PredictionResponse
from model.predict import model,MODEL_VERSION,predict_output
import pickle
import pandas as pd
  
app = FastAPI(title="Loan Payback Prediction API")

# human readable
@app.get("/")
def home():
    return {"message": "Loan Prediction API"}


# machine readable
@app.get('/health')
def health_check():
    return {
        'status': 'OK',
        'version': MODEL_VERSION,
        'model_loaded': model is not None}


# predict
@app.post("/predict",response_model=PredictionResponse)
def loan_prediction(data:UserInput):
    
    user_input = data.model_dump()
    
    try:
        result = predict_output(user_input)
        return result
    
    except Exception as e:
        return JSONResponse(status_code=500, content=str(e))
        
    
    
