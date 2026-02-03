from pydantic import BaseModel,Field
from typing import Annotated,Literal

class PredictionResponse(BaseModel):
    loan_paid_back_probability: float = Field(
        ..., description="Predicted probability that the loan will be paid back"
    )
    loan_paid_back_prediction: int = Field(
        ..., description="0 = Not Paid Back, 1 = Paid Back"
    )