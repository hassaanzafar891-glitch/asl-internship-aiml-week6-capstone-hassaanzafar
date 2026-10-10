"""
Capstone demo: NYC taxi payment-method predictor.

Streamlit interactive demo wrapping the tuned Random Forest model built
across Weeks 3-5 of the internship. The model predicts whether a taxi
ride was most likely paid for with a credit card or cash, based on trip
details.

Run locally with:
    streamlit run app.py
"""

import streamlit as st

from model_service import InvalidInputError, ModelService, log_prediction

st.set_page_config(page_title="Taxi Payment Predictor", page_icon="\U0001F695")


@st.cache_resource
def get_model_service() -> ModelService:
    service = ModelService()
    service.load()
    return service


st.title("NYC Taxi Payment Predictor")
st.write(
    "This demo wraps the tuned Random Forest model built during the "
    "Advance Soft Logics AI/ML internship (Weeks 3-5). Enter trip details "
    "and it predicts whether the ride was most likely paid by **credit card** "
    "or **cash**, along with the model's confidence."
)

try:
    service = get_model_service()
except FileNotFoundError as exc:
    st.error(
        "Model file not found. Run `python scripts/train_and_save_model.py` "
        "first to generate it."
    )
    st.stop()

st.caption(f"Model version: `{service.version}`")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        passengers = st.number_input("Passengers", min_value=1, max_value=6, value=1, step=1)
        distance = st.number_input("Distance (miles)", min_value=0.0, value=2.5, step=0.1)
        fare = st.number_input("Fare ($)", min_value=0.0, value=12.0, step=0.5)
        tolls = st.number_input("Tolls ($)", min_value=0.0, value=0.0, step=0.5)

    with col2:
        color = st.selectbox("Taxi color", ["yellow", "green"])
        pickup_borough = st.selectbox(
            "Pickup borough",
            ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR"],
        )
        dropoff_borough = st.selectbox(
            "Dropoff borough",
            ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR"],
            index=0,
        )

    submitted = st.form_submit_button("Predict payment method")

if submitted:
    record = {
        "passengers": int(passengers),
        "distance": float(distance),
        "fare": float(fare),
        "tolls": float(tolls),
        "color": color,
        "pickup_borough": pickup_borough,
        "dropoff_borough": dropoff_borough,
    }
    try:
        result = service.predict(record)
    except InvalidInputError as exc:
        st.error(f"Invalid input: {exc}")
    else:
        log_prediction(record, result)
        st.success(f"Predicted payment method: **{result['prediction']}**")
        if result["probability"] is not None:
            st.metric("Model confidence", f"{result['probability'] * 100:.1f}%")

st.divider()
st.caption(
    "Model: Random Forest Classifier · trained on the NYC Taxis dataset "
    "(seaborn-data) · predicting `payment` from trip details. "
    "Part of the Advance Soft Logics AI/ML internship, Week 6 capstone."
)
