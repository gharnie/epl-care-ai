
# EPL Care AI - Model Card

## What This Model Does
Predicts whether a patient who comes in for post-pregnancy loss care will 
receive a follow-up appointment. The output helps health workers identify 
patients at risk of falling through the gaps before they leave the facility.

## Training Data
Kenya Post-Abortion Care Survey, 2012.
3,167 patient records across multiple facility types and provinces.
Facility-level data from 328 facilities merged by geography.

## Performance
Accuracy: 84.70%
F1 Score: 85.96%
ROC-AUC: 90.27%
Cross Validation AUC: 93.31%

## What Drives The Predictions
The five factors that influence the model most are:
- Whether the patient was referred for further care
- How long the patient spent in the facility
- The clinical diagnosis
- The reason for the visit
- Whether the patient delayed seeking care beyond 12 weeks

## Limitations
- Data is from Kenya 2012 and may not reflect current conditions
- No post-discharge follow-up data was available for training
- Mental health outcomes were not captured
- Facility matching was done by geography, not by exact facility ID
- Should be retrained with more recent data before live deployment

## How It Should Be Used
- To support clinical decisions, not replace them
- To flag patients for proactive follow-up before discharge
- To identify systemic gaps in service delivery by facility and region
- Never to deny or delay care to any patient
