# app.py
# =========================================
# E-COMMERCE CUSTOMER CHURN PREDICTION APP
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
    page_title="E-Commerce Churn Prediction",
    page_icon="📉",
    layout="centered"
)

# =========================================
# TITLE
# =========================================

st.title("📉 E-Commerce Customer Churn Prediction")
st.markdown(
    "Predict whether an e-commerce customer is likely to churn using Machine Learning."
)

st.divider()

# =========================================
# INPUT FORM
# =========================================

st.subheader("📋 Customer Information")

with st.form("prediction_form"):

    st.markdown("### Numeric Features")

    tenure = st.number_input(
        "Tenure",
        min_value=0,
        max_value=100,
        value=12
    )

    city_tier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    warehouse_to_home = st.number_input(
        "Warehouse To Home Distance",
        min_value=0,
        max_value=200,
        value=10
    )

    hour_spend_on_app = st.number_input(
        "Hour Spend On App",
        min_value=0.0,
        max_value=24.0,
        value=3.0
    )

    number_of_device_registered = st.number_input(
        "Number Of Device Registered",
        min_value=1,
        max_value=10,
        value=3
    )

    satisfaction_score = st.selectbox(
        "Satisfaction Score",
        [1, 2, 3, 4, 5]
    )

    number_of_address = st.number_input(
        "Number Of Address",
        min_value=1,
        max_value=30,
        value=2
    )

    complain = st.selectbox(
        "Complain",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    order_amount_hike = st.number_input(
        "Order Amount Hike From Last Year",
        min_value=0,
        max_value=100,
        value=15
    )

    coupon_used = st.number_input(
        "Coupon Used",
        min_value=0,
        max_value=100,
        value=1
    )

    order_count = st.number_input(
        "Order Count",
        min_value=0,
        max_value=100,
        value=5
    )

    day_since_last_order = st.number_input(
        "Day Since Last Order",
        min_value=0,
        max_value=365,
        value=7
    )

    cashback_amount = st.number_input(
        "Cashback Amount",
        min_value=0.0,
        max_value=10000.0,
        value=150.0
    )

    st.markdown("### Categorical Features")

    preferred_login_device = st.selectbox(
        "Preferred Login Device",
        [
            "Computer",
            "Mobile Phone",
            "Phone"
        ]
    )
    preferred_payment_mode = st.selectbox(
        "Preferred Payment Mode",
        [
            "COD",
            "Cash on Delivery",
            "Credit Card",
            "Debit Card",
            "E wallet",
            "UPI"
        ]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    preferred_order_cat = st.selectbox(
        "Preferred Order Category",
        [
            "Fashion",
            "Grocery",
            "Laptop & Accessory",
            "Mobile",
            "Mobile Phone",
            "Others"
        ]
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Divorced",
            "Married",
            "Single"
        ]
    )

    submit_button = st.form_submit_button("🔍 Predict Churn")

# =========================================
# PREDICTION
# =========================================

    if submit_button:

        input_data = {
        # Numeric features
        "Tenure": tenure,
        "CityTier": city_tier,
        "WarehouseToHome": warehouse_to_home,
        "HourSpendOnApp": hour_spend_on_app,
        "NumberOfDeviceRegistered": number_of_device_registered,
        "SatisfactionScore": satisfaction_score,
        "NumberOfAddress": number_of_address,
        "Complain": complain,
        "OrderAmountHikeFromlastYear": order_amount_hike,
        "CouponUsed": coupon_used,
        "OrderCount": order_count,
        "DaySinceLastOrder": day_since_last_order,
        "CashbackAmount": cashback_amount,

        # One-hot encoded features
        "PreferredLoginDevice_Mobile Phone":
            1 if preferred_login_device == "Mobile Phone" else 0,

        "PreferredLoginDevice_Phone":
            1 if preferred_login_device == "Phone" else 0,

        "PreferredPaymentMode_COD":
            1 if preferred_payment_mode == "COD" else 0,

        "PreferredPaymentMode_Cash on Delivery":
            1 if preferred_payment_mode == "Cash on Delivery" else 0,

        "PreferredPaymentMode_Credit Card":
            1 if preferred_payment_mode == "Credit Card" else 0,

        "PreferredPaymentMode_Debit Card":
            1 if preferred_payment_mode == "Debit Card" else 0,

        "PreferredPaymentMode_E wallet":
            1 if preferred_payment_mode == "E wallet" else 0,

        "PreferredPaymentMode_UPI":
            1 if preferred_payment_mode == "UPI" else 0,

        "Gender_Male":
            1 if gender == "Male" else 0,

        "PreferedOrderCat_Grocery":
            1 if preferred_order_cat == "Grocery" else 0,

        "PreferedOrderCat_Laptop & Accessory":
            1 if preferred_order_cat == "Laptop & Accessory" else 0,

        "PreferedOrderCat_Mobile":
            1 if preferred_order_cat == "Mobile" else 0,

        "PreferedOrderCat_Mobile Phone":
            1 if preferred_order_cat == "Mobile Phone" else 0,

        "PreferedOrderCat_Others":
            1 if preferred_order_cat == "Others" else 0,
        "MaritalStatus_Married":
            1 if marital_status == "Married" else 0,

        "MaritalStatus_Single":
            1 if marital_status == "Single" else 0,
        }

        result = predict_churn(input_data)

        prediction = result["prediction"]
        probability = result["probability"]
        label = result["label"]

        st.divider()

        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.error("⚠️ Customer is likely to CHURN")
        else:
            st.success("✅ Customer is likely to STAY")

        st.metric(
            label="Churn Probability",
            value=f"{probability * 100:.2f}%"
        )

        st.progress(float(probability))

        with st.expander("📄 View Model Input Data"):
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
    - CSV columns should match the model feature names.
    - Missing columns will be filled with `0` automatically.
    - Extra columns will be ignored automatically.
    """
)

st.code(
    """
    Tenure,CityTier,WarehouseToHome,HourSpendOnApp,NumberOfDeviceRegistered,SatisfactionScore,NumberOfAddress,Complain,OrderAmountHikeFromlastYear,CouponUsed,OrderCount,DaySinceLastOrder,CashbackAmount,PreferredLoginDevice_Mobile Phone,PreferredLoginDevice_Phone,PreferredPaymentMode_COD,PreferredPaymentMode_Cash on Delivery,PreferredPaymentMode_Credit Card,PreferredPaymentMode_Debit Card,PreferredPaymentMode_E wallet,PreferredPaymentMode_UPI,Gender_Male,PreferedOrderCat_Grocery,PreferedOrderCat_Laptop & Accessory,PreferedOrderCat_Mobile,PreferedOrderCat_Mobile Phone,PreferedOrderCat_Others,MaritalStatus_Married,MaritalStatus_Single
    12,1,10,3,3,4,2,0,15,1,5,7,150,1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,1
    6,2,15,4,4,3,3,1,20,2,8,3,220,0,1,1,0,0,0,0,0,0,1,0,1,0,0,1,0
    24,3,8,2,2,5,4,0,10,5,12,1,320,1,0,0,1,0,1,0,0,1,0,0,0,1,0,0,1""",
    language="csv"
)

uploaded_file = st.file_uploader(
    "Upload customer CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        batch_df = pd.read_csv(uploaded_file)

        st.write("Preview data:")
        st.dataframe(batch_df.head())

        if st.button("🚀 Predict CSV"):

            results = []

            for _, row in batch_df.iterrows():
                customer_data = row.to_dict()
                prediction_result = predict_churn(customer_data)

                results.append({
                    "Prediction": prediction_result["label"],
                    "Churn Probability": prediction_result["probability"]
                })

                results_df = pd.concat(
                [
                    batch_df.reset_index(drop=True),
                    pd.DataFrame(results)
                ],
                axis=1
            )

            st.success("Prediction completed!")
            st.dataframe(results_df)

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

st.caption("Machine Learning E-Commerce Customer Churn Prediction App")