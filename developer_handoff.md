
# Developer Handoff - EPL Care AI

## What The Model Does
Takes patient information entered through a form and returns a risk score 
showing whether the patient is likely to receive follow-up care after 
early pregnancy loss treatment.

## Files You Need
- epl_model.pkl
- epl_scaler.pkl

## Setup
Python 3.11 with pandas, scikit-learn, and joblib installed.

## Loading The Model
import joblib
import pandas as pd

model = joblib.load('epl_model.pkl')
scaler = joblib.load('epl_scaler.pkl')

## Form Fields
Collect these fields from the health worker on the website form:

Patient age (number)
Location - Urban or Rural
Marital status
Education level
Occupation
Religion
Reason for visit
Gestational age in weeks (number)
Previous abortion history - Yes or No
Came with referral note - Yes or No
Gestational category - 12 weeks or under, or over 12 weeks
Clinical diagnosis
Complications present - Yes or No
Uterine evacuation performed - Yes or No
Procedure type
Who performed the procedure
Anaesthesia given - Yes or No
Pain medication given - Yes or No
Referred for further care - Yes or No
Partner involved in counselling - Yes or No
Post-procedure counselling done - Yes or No
Contraceptive method accepted - Yes or No
Discharge outcome
Time spent in facility
Province, county, district, facility type

## What The Model Returns
prediction: 1 means likely to get follow-up, 0 means unlikely
probability: a number between 0 and 1 showing confidence
risk_level: Low Risk, Medium Risk, or High Risk

## How To Display Results
Low Risk - green banner - patient likely to complete care
Medium Risk - orange banner - monitor this patient closely  
High Risk - red banner - health worker should intervene immediately

## Charts For The Dashboard
Six charts are included in the project folder showing:
- Age distribution of EPL patients
- Follow-up rates by location
- Follow-up rates by education level
- Follow-up rates by facility type
- Follow-up rates by clinical diagnosis
- Follow-up rates by referral status

These can be displayed on an analytics dashboard alongside the prediction tool.

## Important
Always show a disclaimer on the website that the model supports 
clinical decisions but does not replace the judgement of a health worker.
All patient data must be stored and handled securely.
