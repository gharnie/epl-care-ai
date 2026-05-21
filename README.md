
# EPL Care AI

An AI model that predicts whether a patient experiencing early pregnancy loss will receive complete post-loss care including a follow-up appointment.

## What This Project Does
- Analyses patient, facility, and provider data from post-abortion care surveys
- Trains a Random Forest classifier to predict follow-up appointment completion
- Identifies patients at risk of falling through gaps in care
- Provides a REST API for integration with a healthcare website

## Model Performance
- Accuracy: 84.70%
- F1 Score: 85.96%
- ROC-AUC: 90.27%
- Cross Validation AUC: 93.31%

## Project Structure
- epl_analysis.ipynb — full data science pipeline
- api.py — FastAPI REST API
- epl_model.pkl — trained Random Forest model
- epl_scaler.pkl — data scaler
- feature_columns.pkl — model feature columns
- model_card.md — model documentation
- developer_handoff.md — developer integration guide
- charts — EDA visualisations

## Running The API
Install dependencies:
pip install fastapi uvicorn pandas scikit-learn joblib

Run the API:
uvicorn api:app --reload

The API will be live at http://127.0.0.1:8000
API documentation at http://127.0.0.1:8000/docs

## Data
Kenya Post-Abortion Care Survey 2012
- 3,167 patient records
- 328 facility records
- 124 provider records

## Built For
Hackathon: Early Pregnancy Loss Care AI Solutions
Topic: Improving service delivery, referral pathways, and equitable access to post-pregnancy loss care
