import pickle
import pandas as pd

# import the ml model
with open("model/model.pkl","rb") as f:
    model = pickle.load(f)
    
# mlflow
MODEL_VERSION = "1.0.0"

THRESOLD = 0.5

def predict_output(user_input: dict) -> dict:

    input_df = pd.DataFrame([user_input])
    
    # predict probability of loan_paid_back = 1
    prob_paid = model.predict_proba(input_df)[:, 1][0]

    # apply threshold
    prediction = 1 if prob_paid >= THRESOLD else 0

    return {"loan_paid_back_probability": float(round(prob_paid, 4)),
            "loan_paid_back_prediction": int(prediction)}