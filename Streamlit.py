import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model
model = joblib.load('model.pkl')


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




def decode_categorical_variables(data, decoders):
    decoded_data = {}
    for variable, decoder in decoders.items():
        if variable in data:
            decoded_data[variable] = decoder[data[variable]]
    for variable in data:
        if variable not in decoders:
            decoded_data[variable] = data[variable]
    return decoded_data





# Streamlit app UI
st.title('Credit Risk Probability Calculator')

# Add input widgets for user input
st.sidebar.header('Input Features')

person_income = st.sidebar.number_input('Monthly Income', min_value=0.0, value=5000.0, step=100.0)
person_home_ownership = st.selectbox("Home Ownership", ('MORTGAGE', 'OTHER', 'OWN', 'RENT'))
person_emp_length = st.sidebar.number_input('Number of Times 30-59 Days Past Due Not Worse', min_value=0, max_value=150, value=0)
loan_intent = st.selectbox("Loan Intent", ('DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE'))
loan_grade = st.selectbox("Loan Grade", ('A', 'B', 'C', 'D', 'E', 'F', 'G'))
loan_amnt = st.sidebar.number_input('Loan Amount', min_value=0, max_value=50, value=5)
loan_int_rate = st.sidebar.number_input('Loan Interest Rate', min_value=0, max_value=50, value=0)
loan_percent_income = st.sidebar.number_input('Number of Real Estate Loans or Lines', min_value=0, max_value=1, value=0)
cb_person_default_on_file = st.selectbox("Default History", ('N', 'Y'))
cb_person_cred_hist_length = st.sidebar.number_input('Number of Dependents', min_value=0, max_value=50, value=1)

# Create a button to trigger prediction

predict_button = st.sidebar.button('Predict')

# Display prediction result
if predict_button:
    user_input = {
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length
    }

    # Decode categorical variables
    decoded_data = decode_categorical_variables(user_input, decoders)

    if (
        "person_home_ownership" not in decoded_data
        or "loan_intent" not in decoded_data
        or "loan_grade" not in decoded_data
        or "cb_person_default_on_file" not in decoded_data
    ):
        st.write('Invalid input')
    else:
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

        # Make prediction
        predicted_prob = model.predict_proba(input_data)[0][1]
        st.subheader('Prediction Probability')
        st.write(f'Probability of Credit Risk: {predicted_prob:.4f}')

        # Interpretation
        if predicted_prob >= 0.5:
            st.write('Interpretation: Credit Risk')
        else:
            st.write('Interpretation: No Credit Risk')

# Run the Streamlit app
if __name__ == '__main__':
    st.sidebar.title('About')
    st.sidebar.info('This app is a Credit Risk Probability Calculator.')