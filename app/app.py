import streamlit as st
import pandas as pd

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
        ["1", "2", "3"]
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
        "Gender": [gender],
        "Tenure": [tenure],
        "PreferredLoginDevice": [preferred_login_device],
        "CityTier": [city_tier],
        "SatisfactionScore": [satisfaction],
        "MaritalStatus": [marriage],
        "NumberOfAddress": [number_of_addresses],
        "WarehouseToHome": [warehouse_to_home],
        "HourSpendOnApp": [hour_spend],
        "PreferredPaymentMode": [preferred_payment_mode],
        "NumberOfDeviceRegistered": [device_registration],
        "PreferedOrderCat": [preferred_order_cat],
        "Complaint": [complaint],
        "OrderAmountHikeFromlastYear": [order_amount],
        "CouponUsed": [used_coupon],
        "OrderCount": [order_count],
        "DaySinceLastOrder": [day_since_last_order],
        "CashbackAmount": [cash_back_amount]
    })

    # Dummy prediction logic
    # Thay bằng model.predict() sau này

    risk_score = 0

    if tenure < 6:
        risk_score += 1

    if satisfaction <= 2:
        risk_score += 1

    if complaint > 2:
        risk_score += 1

    if day_since_last_order > 20:
        risk_score += 1

    if risk_score >= 2:
        prediction = 1
        probability = 0.84
    else:
        prediction = 0
        probability = 0.16

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠️ Customer is likely to CHURN\n\n"
            f"Probability: {probability:.2%}"
        )
    else:
        st.success(
            f"✅ Customer is likely to STAY\n\n"
            f"Probability of churn: {probability:.2%}"
        )

    st.subheader("Input Data")

    st.dataframe(input_data)