from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load('epl_model.pkl')
scaler = joblib.load('epl_scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

class PatientData(BaseModel):
    province: str = "Unknown"
    county: str = "Unknown"
    district: str = "Unknown"
    type: str = "Health Centre"
    pds101: float = 25.0
    pds102: str = "Urban"
    pds103: str = "Married"
    pds104: str = "Complete Secondary"
    pds105: str = "Other Christian"
    pds106: str = "Farming"
    pds208: str = "Yes, wanted then"
    pds301: str = "Postabortion Care"
    pds302: float = 8.0
    pds303: str = "No"
    pds310: str = "No"
    pds324: str = "<=12 weeks"
    pds401: str = "Incomplete Abortion"
    pds402: str = "No"
    pds501: str = "Yes"
    pds502: str = "MVA"
    pds503: str = "Clinical Officer"
    pds505: str = "Yes"
    pds507: str = "Yes"
    pds509: str = "No"
    pds510: str = "No"
    pds701: str = "Yes"
    pds702: str = "Yes"
    pds801: str = "Discharged well"
    pds802: str = "Less than 12 Hrs"
    ses_score: float = 3.0
    mental_health_risk: int = 0
    care_delay: int = 0
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

def identify_care_gaps(data):
    gaps = []

    if data.pds509 == "No":
        gaps.append("Patient was not referred for further care — referral is the strongest predictor of follow-up completion")

    if data.pds510 == "No":
        gaps.append("Male partner was not included in post-loss counselling — reduces likelihood of follow-up attendance and shared care decisions")

    if data.pds701 == "No":
        gaps.append("Post-procedure counselling was not completed — patient left without emotional support or care instructions")

    if data.pds507 == "No":
        gaps.append("Pain medication was not given — basic pain management was missed")

    if data.pds310 == "No":
        gaps.append("Patient arrived without a referral note — suggests breakdown in the referral pathway before reaching this facility")

    if data.care_delay == 1:
        gaps.append("Patient presented after 12 weeks — delayed care seeking indicates access barriers or lack of awareness")

    if data.pds702 == "No":
        gaps.append("Patient did not accept contraceptive counselling — post-loss family planning discussion was incomplete")

    if data.pds802 in ["Less than 12 Hrs", "Less than 12 hours"]:
        gaps.append("Patient spent less than 12 hours in the facility — rushed discharge increases risk of incomplete care")

    if data.pds801 in ["Left against advice", "Left Against Medical Advice"]:
        gaps.append("Patient left against medical advice — high risk of not completing post-loss care")

    if data.pds503 in ["Unknown", "Other"]:
        gaps.append("Procedure was performed by an unidentified or unqualified provider — care quality uncertain")

    if data.pds401 in ["Other", "Unknown"]:
        gaps.append("Clinical diagnosis was unclassified — unusual cases are most likely to fall through care gaps")

    return gaps


def identify_equity_flags(data):
    flags = []

    if data.pds102 == "Rural":
        flags.append("Rural location — patient faces geographic and transport barriers to follow-up care")

    if data.pds104 in ["No education", "Incomplete Primary", "Complete Primary"]:
        flags.append("Low education level — patient may need additional support navigating the healthcare system")

    if data.ses_score <= 2:
        flags.append("Very low socioeconomic status — financial barriers may prevent follow-up attendance")

    elif data.ses_score <= 4:
        flags.append("Below average socioeconomic status — cost of follow-up care may be a barrier")

    if data.hf407 in ["No", "Unknown"]:
        flags.append("This facility has no formal referral system — patients are more likely to be lost between facilities")

    if data.hf405 in ["No", "Unknown"]:
        flags.append("This facility has no written PAC protocol — care consistency and quality may be compromised")

    if data.hf303 in ["No", "Sometimes"]:
        flags.append("This facility does not consistently provide counselling to all PAC patients — service delivery gap at facility level")

    if data.pds302 > 12:
        flags.append("Second trimester loss — patient requires a more intensive care pathway and closer follow-up")

    if data.pds510 == "No" and data.pds103 == "Married":
        flags.append("Married patient whose partner was excluded from counselling — partner support gap increases drop-out risk")

    return flags


def get_action(risk_level):
    if risk_level == "High Risk":
        return "Do not discharge this patient without a confirmed follow-up appointment. Intervene immediately. Assign a community health worker for post-discharge support. Conduct mental health screening before discharge."
    elif risk_level == "Medium Risk":
        return "Confirm follow-up appointment is scheduled before discharge. Call or text patient within 48 hours to confirm attendance. Flag for community health worker check-in."
    else:
        return "Proceed with standard discharge. Ensure follow-up appointment is scheduled and documented in patient records."


def get_followup_recommendation(data, risk_level):
    base = ""

    if risk_level == "High Risk":
        base = "Schedule follow-up appointment before the patient leaves the facility. Do not allow discharge without a confirmed date and contact number."
    elif risk_level == "Medium Risk":
        base = "Schedule follow-up appointment within 1 week. Contact patient within 48 hours to confirm attendance."
    else:
        base = "Schedule standard follow-up appointment within 2 weeks. Document in patient records."

    if data.pds510 == "No" and data.pds103 in ["Married", "Living together"]:
        base += " Consider arranging a partner counselling session at the follow-up visit to improve household support for the patient's recovery."

    if data.mental_health_risk == 1:
        base += " Refer patient to mental health or grief counselling service before or at the follow-up appointment."

    if data.pds102 == "Rural":
        base += " Coordinate with community health worker for home visit given patient's rural location."

    if data.care_delay == 1:
        base += " Patient delayed seeking care — extra outreach may be needed to ensure follow-up attendance."

    return base


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
            try:
                input_df[col] = input_df[col].astype('category').cat.codes
            except Exception:
                input_df[col] = 0

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if probability >= 0.7:
        risk_level = "Low Risk"
    elif probability >= 0.4:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    care_gaps = identify_care_gaps(data)
    equity_flags = identify_equity_flags(data)
    action = get_action(risk_level)
    followup_recommendation = get_followup_recommendation(data, risk_level)

    mental_health_flag = data.mental_health_risk == 1
    mental_health_note = (
        "Patient has mental health risk factors. Psychological support and grief counselling recommended."
        if mental_health_flag else
        "No immediate mental health risk factors identified. Standard post-loss support applies."
    )

    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 4),
        "risk_level": risk_level,
        "action": action,
        "care_gaps": care_gaps,
        "mental_health_flag": mental_health_flag,
        "mental_health_note": mental_health_note,
        "equity_flags": equity_flags,
        "follow_up_recommendation": followup_recommendation
    }