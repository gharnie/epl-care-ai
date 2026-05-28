from flask import Flask, request, jsonify
import pandas as pd
import joblib

import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, 'epl_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'epl_scaler.pkl'))
feature_columns = joblib.load(os.path.join(BASE_DIR, 'feature_columns.pkl'))

def identify_care_gaps(data):
    gaps = []
    if data.get('pds509') == "No":
        gaps.append("Patient was not referred for further care — referral is the strongest predictor of follow-up completion")
    if data.get('pds510') == "No":
        gaps.append("Male partner was not included in post-loss counselling — reduces likelihood of follow-up attendance")
    if data.get('pds701') == "No":
        gaps.append("Post-procedure counselling was not completed — patient left without emotional support")
    if data.get('pds507') == "No":
        gaps.append("Pain medication was not given — basic pain management was missed")
    if data.get('pds310') == "No":
        gaps.append("Patient arrived without a referral note — breakdown in referral pathway")
    if data.get('care_delay') == 1:
        gaps.append("Patient presented after 12 weeks — delayed care seeking indicates access barriers")
    if data.get('pds702') == "No":
        gaps.append("Patient did not accept contraceptive counselling — post-loss family planning incomplete")
    if data.get('pds802') in ["Less than 12 Hrs", "Less than 12 hours"]:
        gaps.append("Patient spent less than 12 hours in the facility — rushed discharge increases risk")
    if data.get('pds801') in ["Left against advice", "Left Against Medical Advice"]:
        gaps.append("Patient left against medical advice — high risk of not completing post-loss care")
    if data.get('pds503') in ["Unknown", "Other"]:
        gaps.append("Procedure performed by unidentified provider — care quality uncertain")
    if data.get('pds401') in ["Other", "Unknown"]:
        gaps.append("Clinical diagnosis unclassified — unusual cases most likely to fall through care gaps")
    return gaps


def identify_equity_flags(data):
    flags = []
    if data.get('pds102') == "Rural":
        flags.append("Rural location — patient faces geographic and transport barriers to follow-up care")
    if data.get('pds104') in ["No education", "Incomplete Primary", "Complete Primary"]:
        flags.append("Low education level — patient may need extra support navigating the healthcare system")
    if data.get('ses_score', 5) <= 2:
        flags.append("Very low socioeconomic status — financial barriers may prevent follow-up attendance")
    elif data.get('ses_score', 5) <= 4:
        flags.append("Below average socioeconomic status — cost of follow-up care may be a barrier")
    if data.get('hf407') in ["No", "Unknown"]:
        flags.append("Facility has no formal referral system — patients more likely to be lost between facilities")
    if data.get('hf405') in ["No", "Unknown"]:
        flags.append("Facility has no written PAC protocol — care consistency may be compromised")
    if data.get('hf303') in ["No", "Sometimes"]:
        flags.append("Facility does not consistently provide counselling to all PAC patients")
    if data.get('pds302', 0) > 12:
        flags.append("Second trimester loss — patient requires more intensive care pathway")
    if data.get('pds510') == "No" and data.get('pds103') == "Married":
        flags.append("Married patient whose partner was excluded from counselling — increases drop-out risk")
    return flags


def get_action(risk_level):
    if risk_level == "High Risk":
        return "Do not discharge this patient without a confirmed follow-up appointment. Intervene immediately. Assign a community health worker for post-discharge support. Conduct mental health screening before discharge."
    elif risk_level == "Medium Risk":
        return "Confirm follow-up appointment is scheduled before discharge. Call or text patient within 48 hours to confirm attendance. Flag for community health worker check-in."
    else:
        return "Proceed with standard discharge. Ensure follow-up appointment is scheduled and documented in patient records."


def get_followup_recommendation(data, risk_level):
    if risk_level == "High Risk":
        base = "Schedule follow-up appointment before the patient leaves the facility. Do not allow discharge without a confirmed date and contact number."
    elif risk_level == "Medium Risk":
        base = "Schedule follow-up appointment within 1 week. Contact patient within 48 hours to confirm attendance."
    else:
        base = "Schedule standard follow-up appointment within 2 weeks. Document in patient records."

    if data.get('pds510') == "No" and data.get('pds103') in ["Married", "Living together"]:
        base += " Consider arranging a partner counselling session at the follow-up visit to improve household support for the patient's recovery."
    if data.get('mental_health_risk') == 1:
        base += " Refer patient to mental health or grief counselling service before or at the follow-up appointment."
    if data.get('pds102') == "Rural":
        base += " Coordinate with community health worker for home visit given patient's rural location."
    if data.get('care_delay') == 1:
        base += " Patient delayed seeking care — extra outreach may be needed to ensure follow-up attendance."
    return base


@app.route('/')
def home():
    return jsonify({"message": "EPL Care AI API is running"})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No data received. Please send patient data as JSON."
            }), 400

        defaults = {
            "province": "Central", "county": "Kiambu", "district": "Kiambu East",
            "type": "Health Centre", "pds101": 25.0, "pds102": "Urban",
            "pds103": "Married", "pds104": "Complete Secondary", "pds105": "Other Christian",
            "pds106": "Farming", "pds208": "Yes, wanted then", "pds301": "Postabortion Care",
            "pds302": 8.0, "pds303": "No", "pds310": "No", "pds324": "<=12 weeks",
            "pds401": "Incomplete Abortion", "pds402": "No", "pds501": "Yes",
            "pds502": "MVA", "pds503": "Clinical Officer", "pds505": "Yes",
            "pds507": "Yes", "pds509": "No", "pds510": "No", "pds701": "Yes",
            "pds702": "Yes", "pds801": "Discharged well", "pds802": "Less than 12 Hrs",
            "ses_score": 3.0, "mental_health_risk": 0, "care_delay": 0,
            "hf215": "Unknown", "hf303": "Unknown", "hf305a": "Unknown",
            "hf308": "Unknown", "hf310a": "Unknown", "hf401": "Unknown",
            "hf402a": "Unknown", "hf405": "Unknown", "hf407": "Unknown",
            "pac_jan": 0.0, "pac_feb": 0.0, "pac_mar": 0.0, "pac_apr": 0.0,
            "pac_may": 0.0, "pac_jun": 0.0, "pac_jul": 0.0, "pac_aug": 0.0,
            "pac_sep": 0.0, "pac_oct": 0.0, "pac_nov": 0.0, "pac_dec": 0.0
        }

        for key, value in defaults.items():
            if key not in data or data[key] is None or data[key] == "":
                data[key] = value

        for key in list(data.keys()):
            if key not in defaults and key not in feature_columns:
                data.pop(key)

        input_df = pd.DataFrame([data])
        input_df = input_df[feature_columns]

        for col in input_df.columns:
            if input_df[col].dtype == 'object':
                try:
                    input_df[col] = input_df[col].astype('category').cat.codes
                except Exception:
                    input_df[col] = 0

        for col in input_df.columns:
            try:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
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
        mental_health_flag = data.get('mental_health_risk') == 1
        mental_health_note = (
            "Patient has mental health risk factors. Psychological support and grief counselling recommended."
            if mental_health_flag else
            "No immediate mental health risk factors identified. Standard post-loss support applies."
        )

        return jsonify({
            "prediction": int(prediction),
            "probability": round(float(probability), 4),
            "risk_level": risk_level,
            "action": action,
            "care_gaps": care_gaps,
            "mental_health_flag": mental_health_flag,
            "mental_health_note": mental_health_note,
            "equity_flags": equity_flags,
            "follow_up_recommendation": followup_recommendation
        })

    except Exception as e:
        return jsonify({
            "error": "Something went wrong processing this request.",
            "details": str(e),
            "prediction": None,
            "probability": None,
            "risk_level": None,
            "action": "Unable to generate prediction. Please check the input data and try again.",
            "care_gaps": [],
            "mental_health_flag": None,
            "mental_health_note": None,
            "equity_flags": [],
            "follow_up_recommendation": "Please resubmit with valid patient data."
        }), 500


if __name__ == '__main__':
    app.run(debug=True)