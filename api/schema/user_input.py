from pydantic import BaseModel,Field
from typing import Annotated,Literal

class UserInput(BaseModel):

    annual_income: Annotated[float,Field(...,gt=0,description="Applicant's annual income")]
    
    debt_to_income_ratio: Annotated[float,Field(...,ge=0,le=1,description="Debt-to-income ratio as a percentage")]
    
    credit_score:Annotated[int,Field(...,ge=300,le=850,description="Applicant's credit score")]
    
    loan_amount:Annotated[float,Field(...,gt=0,description="Loan amount requested" )]
    
    interest_rate:Annotated[float, Field( ...,ge=0,le=100,description="Interest rate percentage")]
    
    gender:Annotated[Literal["Female", "Male", "Other"],Field(..., description="Gender of the applicant")]
    
    marital_status:Annotated[Literal["Single", "Married", "Divorced", "Widowed"],Field(...,description="Marital status of the applicant")]
    
    education_level:Annotated[Literal["High School", "Master's", "Bachelor's", "PhD", "Other"],Field(...,description="Highest education qualification")]
    
    employment_status:Annotated[Literal["Self-employed", "Employed", "Unemployed", "Retired", "Student"],Field(...,description="Current employment status")] 
    
    loan_purpose:Annotated[Literal["Other","Debt consolidation","Home","Education",
            "Vacation","Car","Medical","Business"],Field(...,description="Purpose of the loan")]
    
    grade_subgrade:Annotated[ Literal["A1","A2","A3","A4","A5",
            "B1","B2","B3","B4","B5",
            "C1","C2","C3","C4","C5",
            "D1","D2","D3","D4","D5",
            "E1","E2","E3","E4","E5",
            "F1","F2","F3","F4","F5"
        ],Field(...,description="Loan grade and subgrade assigned by lender")]
    
    
    