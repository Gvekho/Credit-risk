from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load your pre-trained model using joblib
model = joblib.load("model.pkl")

# Define the decoders
decoders = {
    "person_home_ownership": {
        "MORTGAGE": 0,
        "OTHER": 1,
        "OWN": 2,
        "RENT": 3
    },
    "loan_intent": {
        "DEBTCONSOLIDATION": 0,
        "EDUCATION": 1,
        "HOMEIMPROVEMENT": 2,
        "MEDICAL": 3,
        "PERSONAL": 4,
        "VENTURE": 5
    },
    "loan_grade": {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6
    },
    "cb_person_default_on_file": {
        "N": 0,
        "Y": 1
    }
}

class CreditRiskRequest(BaseModel):
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: int
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int

class CreditRiskResponse(BaseModel):
    prediction: str

def decode_categorical_variables(data, decoders):
    decoded_data = {}
    for variable, decoder in decoders.items():
        if variable in data:
            decoded_data[variable] = decoder[data[variable]]
    for variable in data:
        if variable not in decoders:
            decoded_data[variable] = data[variable]
    return decoded_data

@app.post("/predict_credit_risk/", response_model=CreditRiskResponse)
def predict_credit_risk(data: CreditRiskRequest):
    # Decode categorical variables
    decoded_data = decode_categorical_variables(data.dict(), decoders)

    if (
        "person_home_ownership" not in decoded_data
        or "loan_intent" not in decoded_data
        or "loan_grade" not in decoded_data
        or "cb_person_default_on_file" not in decoded_data
    ):
        return {"prediction": "Invalid input"}

    # Prepare the input data for prediction
    input_data = np.array([
        decoded_data["person_income"],
        decoded_data["person_home_ownership"],
        decoded_data["person_emp_length"],
        decoded_data["loan_intent"],
        decoded_data["loan_grade"],
        decoded_data["loan_amnt"],
        decoded_data["loan_int_rate"],
        decoded_data["loan_percent_income"],
        decoded_data["cb_person_default_on_file"],
        decoded_data["cb_person_cred_hist_length"]
    ]).reshape(1, -1)

    # Make the prediction
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)[0][1]

    # The model's output is a probability; you can set a threshold to classify as "Good" or "Bad"
    threshold = 0.5
    if prediction >= threshold:
        return {"prediction": "Your credit risk probability is {}%, most likely it is Deafult".format(round(proba * 100,2))}
    else:
        return {"prediction": "Your credit risk probability is {}%, most likely it is not Deafult".format(round(proba * 100,2))}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
