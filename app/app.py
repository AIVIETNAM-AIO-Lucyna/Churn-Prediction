# app.py
# =========================================
# CUSTOMER CHURN PREDICTION APP
# STREAMLIT APP
# =========================================

import streamlit as st
import pandas as pd
import sys
import os

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(ROOT_DIR)

from src.predict import predict_churn

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)

# =========================================
# TITLE
# =========================================

st.title("📉 Customer Churn Prediction")
st.markdown(
    "Predicting whether customers will leave a service using Machine Learning"
)

st.divider()

# =========================================
# INPUT FORM
# =========================================

st.subheader("📋 Customer Information")

with st.form("prediction_form"):

    # ==============================
    # NUMERIC FEATURES
    # ==============================

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=120,
        value=12
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=1000.0,
        value=80.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=100000.0,
        value=1000.0
    )

    # ==============================
    # CATEGORICAL FEATURES
    # ==============================

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    # ==============================
    # SUBMIT BUTTON
    # ==============================

    submit_button = st.form_submit_button(
        "🔍 Predict Churn"
    )

# =========================================
# PREDICTION
# =========================================

if submit_button:

    # ======================================
    # BUILD INPUT DATA
    # ======================================

    input_data = {
        "Age": age,
        "Tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,

        # Example encoding
        "Gender_Male": 1 if gender == "Male" else 0,

        "SeniorCitizen": senior_citizen,

        "Partner_Yes": 1 if partner == "Yes" else 0,

        "Dependents_Yes": 1 if dependents == "Yes" else 0,

        "PhoneService_Yes": 1 if phone_service == "Yes" else 0,

        "InternetService_Fiber optic":
            1 if internet_service == "Fiber optic" else 0,

        "InternetService_No":
            1 if internet_service == "No" else 0,

        "Contract_One year":
            1 if contract == "One year" else 0,

        "Contract_Two year":
            1 if contract == "Two year" else 0,

        "PaymentMethod_Credit card (automatic)":
            1 if payment_method == "Credit card (automatic)" else 0,

        "PaymentMethod_Electronic check":
            1 if payment_method == "Electronic check" else 0,

        "PaymentMethod_Mailed check":
            1 if payment_method == "Mailed check" else 0,
    }

    # ======================================
    # PREDICT
    # ======================================

    result = predict_churn(input_data)

    prediction = result["prediction"]
    probability = result["probability"]
    label = result["label"]

    st.divider()

    st.subheader("📊 Prediction Result")

    # ======================================
    # RESULT UI
    # ======================================

    if prediction == 1:

        st.error(
            f"⚠️ Customer is likely to CHURN"
        )

    else:

        st.success(
            f"✅ Customer is likely to STAY"
        )

    st.metric(
        label="Churn Probability",
        value=f"{probability * 100:.2f}%"
    )

    # ======================================
    # PROBABILITY BAR
    # ======================================

    st.progress(float(probability))

    # ======================================
    # SHOW INPUT DATA
    # ======================================

    with st.expander("📄 View Input Data"):

        df = pd.DataFrame([input_data])

        st.dataframe(df)


# =========================================
# BATCH PREDICTION FROM CSV
# =========================================

st.divider()

st.subheader("📂 Batch Prediction From CSV")

st.markdown(
    """
    ### Note:
    - Each row represents one customer.
    - The CSV file must contain columns that match the data used to train the model.
    - The CSV file should be formatted as shown below.
    """
)

st.code(
    """Age,Tenure,MonthlyCharges,TotalCharges,Gender_Male,SeniorCitizen
35,12,80,1000,1,0
42,24,120,2500,0,1""",
    language="csv"
)

uploaded_file = st.file_uploader(
    "Upload customer CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        # ==============================
        # LOAD CSV
        # ==============================

        batch_df = pd.read_csv(uploaded_file)

        st.write("Preview data:")

        st.dataframe(batch_df.head())

        # ==============================
        # PREDICT BUTTON
        # ==============================

        if st.button("🚀 Predict CSV"):

            results = []

            # ==========================
            # LOOP EACH CUSTOMER
            # ==========================

            for _, row in batch_df.iterrows():

                customer_data = row.to_dict()

                prediction = predict_churn(customer_data)

                results.append({
                    "Prediction": prediction["label"],
                    "Probability": prediction["probability"]
                })

            # ==========================
            # ADD RESULT COLUMNS
            # ==========================

            results_df = pd.concat(
                [
                    batch_df.reset_index(drop=True),
                    pd.DataFrame(results)
                ],
                axis=1
            )

            # ==========================
            # SHOW RESULTS
            # ==========================

            st.success("Prediction completed!")

            st.dataframe(results_df)

            # ==========================
            # DOWNLOAD CSV
            # ==========================

            csv = results_df.to_csv(index=False)

            st.download_button(
                label="⬇️ Download Results CSV",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:

        st.error(f"Error: {e}")
# =========================================
# FOOTER
# =========================================

st.divider()

st.caption(
    "Machine Learning Customer Churn Prediction App"
)