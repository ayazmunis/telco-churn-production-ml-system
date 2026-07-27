from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd

from src.preprocessing import preprocess_data
from src.feature_engineering import create_features

app = FastAPI(
    title="Telco Churn Prediction API",
    version="1.0"
)

# Load the trained model when the API starts
model = joblib.load(
    "models/baseline_logistic_regression.pkl"
)

class CustomerData(BaseModel):

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int

    PhoneService: str
    MultipleLines: str
    InternetService: str

    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str

    StreamingTV: str
    StreamingMovies: str

    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def home():
    return {
        "message": "Telco Churn Prediction API is running!"
    }

@app.post("/predict")
def predict(customer: CustomerData):

    input_df = pd.DataFrame(
        [customer.model_dump()]
    )

    # Apply the same feature engineering used during training
    input_df = create_features(input_df)

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": round(
            float(probability),
            4
        ),
        "model_version": "v1.0"
    }