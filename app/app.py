import streamlit as st
import pandas as pd
import sys
import os
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, BASE_DIR)
from src.predict import predict_customer

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.write("Nhập thông tin khách hàng để dự đoán khả năng churn")

st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        options=["Male", "Female"],
    )

    tenure = st.number_input(
        "Tenure",
        min_value=0,
        max_value=100,
        value=12
    )

    preferred_login_device = st.selectbox(
        "Preferred Login Device",
        ["Computer", "Mobile Phone", "Phone"]
    )

    city_tier = st.selectbox(
        "City Tier",
        [1,2,3]
    )

    satisfaction = st.number_input(
        "Satisfaction",
        min_value=0,
        max_value=5,
        value=5
    )

    marriage = st.selectbox(
        "Marriage",
        options=["Single", "Married", "Divorced"]
    )

    number_of_addresses = st.number_input(
        "Number of Addresses",
        min_value=0,
        value=1
    )

    warehouse_to_home = st.number_input(
        "Warehouse To Home",
        min_value=0.0,
        value=10.0
    )

    hour_spend = st.number_input(
        "Hour Spend On App",
        min_value=0.0,
        value=1.0
    )

with col2:

    preferred_payment_mode = st.selectbox(
        "Preferred Payment Mode",
        [
            "Debit Card",
            "Credit Card",
            "E wallet",
            "UPI",
            "CC",
            "COD",
            "Cash on Delivery"
        ]
    )

    device_registration = st.number_input(
        "Device Registration",
        min_value=0.0,
        value=1.0
    )

    preferred_order_cat = st.selectbox(
        "Preferred Order Category",
        options=[
            "Laptop & Accessory",
            "Fashion",
            "Mobile",
            "Mobile Phone",
            "Grocery",
            "Others"
        ]
    )

    complaint = st.number_input(
        "Number of Complaints",
        min_value=0,
        value=0
    )

    order_amount = st.number_input(
        "Order Amount",
        min_value=0.0,
        value=1000.0
    )

    used_coupon = st.number_input(
        "Used Coupon",
        min_value=0,
        value=0
    )

    order_count = st.number_input(
        "Order Count",
        min_value=0,
        value=100
    )

    day_since_last_order = st.number_input(
        "Day Since Last Order",
        min_value=0,
        value=0
    )

    cash_back_amount = st.number_input(
        "Cash Back Amount",
        min_value=0.0,
        value=1000.0
    )

st.divider()

predict_btn = st.button(
    "Predict Churn",
    use_container_width=True
)

if predict_btn:

    input_data = pd.DataFrame({
        "Gender":[gender],
        "Tenure":[tenure],
        "PreferredLoginDevice":[preferred_login_device],
        "CityTier":[int(city_tier)],
        "SatisfactionScore":[satisfaction],
        "MaritalStatus":[marriage],
        "NumberOfAddress":[number_of_addresses],
        "WarehouseToHome":[warehouse_to_home],
        "HourSpendOnApp":[hour_spend],
        "PreferredPaymentMode":[preferred_payment_mode],
        "NumberOfDeviceRegistered":[device_registration],
        "PreferedOrderCat":[preferred_order_cat],

        "Complain":[complaint],

        "OrderAmountHikeFromlastYear":[order_amount],
        "CouponUsed":[used_coupon],
        "OrderCount":[order_count],
        "DaySinceLastOrder":[day_since_last_order],
        "CashbackAmount":[cash_back_amount]
    })
    prediction, probability = predict_customer(
    input_data
)
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠️ Customer is likely to CHURN\n\n"
            f"Probability: {probability}"
        )
    else:
        st.success(
            f"✅ Customer is likely to STAY\n\n"
            f"Probability of churn: {probability}"
        )

    st.subheader("Input Data")

    st.dataframe(input_data)