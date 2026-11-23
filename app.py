from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).parent / "models" / "pizza_order_pipeline.joblib"
PROMOTIONS = {
    "No promotion": "none",
    "Lunch special": "lunch_special",
    "Family bundle": "family_bundle",
    "Percent off": "percent_off",
}
shop_name = st.secrets.get("SHOP_NAME", "Pizza Rush Predictor")


@st.cache_resource
def load_model():
    """Load the fitted notebook pipeline once; never train inside the app."""
    return joblib.load(MODEL_PATH)


def make_prediction_input(selected_date: date, hour: int, promotion: str) -> pd.DataFrame:
    """Build the exact seven-feature row used by the training notebook."""
    timestamp = pd.Timestamp(selected_date)
    return pd.DataFrame(
        [
            {
                "hour": hour,
                "day_of_week": timestamp.day_name(),
                "is_weekend": int(timestamp.dayofweek >= 5),
                "promotion": promotion,
                "month": timestamp.month,
                "day_of_month": timestamp.day,
                "day_of_year": timestamp.dayofyear,
            }
        ]
    )


st.set_page_config(page_title="Pizza Rush Predictor", page_icon="🍕", layout="centered")

st.title(shop_name)
st.write("Choose a time and promotion to estimate how many pizza orders to expect.")

if not MODEL_PATH.exists():
    st.error(
        "The saved model was not found. Run the final ‘Save the complete pipeline’ "
        "cell in `pizza_hourly_order_prediction.ipynb`, then refresh this page."
    )
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"The saved model could not be loaded: {error}")
    st.stop()

with st.form("prediction_form"):
    selected_date = st.date_input("Date", value=date.today())
    selected_hour = st.slider("Hour of day", min_value=0, max_value=23, value=19)
    selected_promotion_label = st.selectbox("Promotion", options=list(PROMOTIONS))
    submitted = st.form_submit_button("Predict orders", type="primary", use_container_width=True)

if submitted:
    input_row = make_prediction_input(
        selected_date,
        selected_hour,
        PROMOTIONS[selected_promotion_label],
    )

    try:
        prediction = max(0, round(float(model.predict(input_row)[0])))
    except Exception as error:
        st.error(f"Prediction failed: {error}")
    else:
        st.metric("Predicted hourly orders", f"{prediction} pizzas")
        st.caption(
            f"Estimate for {selected_date:%A, %d %B %Y} at "
            f"{selected_hour:02d}:00 with {selected_promotion_label.lower()}."
        )
