from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load('epl_model.pkl')
scaler = joblib.load('epl_scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

class PatientData(BaseModel):
    province: str
    county: str
    district: str
    type: str
    pds101: float
    pds102: str
    pds103: str
    pds104: str
    pds105: str
    pds106: str
    pds208: str
    pds301: str
    pds302: float
    pds303: str
    pds310: str
    pds324: str
    pds401: str
    pds402: str
    pds501: str
    pds502: str
    pds503: str
    pds505: str
    pds507: str
    pds509: str
    pds510: str
    pds701: str
    pds702: str
    pds801: str
    pds802: str
    ses_score: float
    mental_health_risk: int
    care_delay: int
    hf215: str = "Unknown"
    hf303: str = "Unknown"
    hf305a: str = "Unknown"
    hf308: str = "Unknown"
    hf310a: str = "Unknown"
    hf401: str = "Unknown"
    hf402a: str = "Unknown"
    hf405: str = "Unknown"
    hf407: str = "Unknown"
    pac_jan: float = 0.0
    pac_feb: float = 0.0
    pac_mar: float = 0.0
    pac_apr: float = 0.0
    pac_may: float = 0.0
    pac_jun: float = 0.0
    pac_jul: float = 0.0
    pac_aug: float = 0.0
    pac_sep: float = 0.0
    pac_oct: float = 0.0
    pac_nov: float = 0.0
    pac_dec: float = 0.0

@app.get("/")
def home():
    return {"message": "EPL Care AI API is running"}

@app.post("/predict")
def predict(data: PatientData):
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_columns]

    for col in input_df.columns:
        if input_df[col].dtype == 'object':
            input_df[col] = input_df[col].astype('category').cat.codes

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if probability >= 0.7:
        risk = "Low Risk"
    elif probability >= 0.4:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 4),
        "risk_level": risk
    }