"""
Streamlit app: predicts whether a customer will purchase the Wellness
Tourism Package. Loads the best model directly from the Hugging Face
model hub at startup.
"""

import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

MODEL_REPO = "VamiSai/visit-with-us-tourism-model"

HF_TOKEN = os.environ.get("HF_TOKEN")


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO, repo_type="model", filename="best_model.joblib", token=HF_TOKEN
    )
    return joblib.load(model_path)


st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🧳")
st.title("🧳 Wellness Tourism Package — Purchase Predictor")
st.write(
    "Enter a prospect's details below to estimate the likelihood they will "
    "purchase the new Wellness Tourism Package."
)

model = load_model()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    num_trips = st.number_input("Avg Number of Trips per Year", min_value=0, max_value=20, value=2)
  

with col2:
    passport = st.selectbox("Has Passport?", ["No", "Yes"])
    own_car = st.selectbox("Owns a Car?", ["No", "Yes"])
    num_children = st.number_input("Number of Children Visiting (<5 yrs)", min_value=0, max_value=5, value=0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=20000)
    pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    num_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=3)
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=15)

if st.button("Predict", type="primary"):
    input_df = pd.DataFrame([{
        "Age": age, "TypeofContact": type_of_contact, "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch, "Occupation": occupation, "Gender": gender,
        "NumberOfPersonVisiting": num_persons, "NumberOfFollowups": num_followups,
        "ProductPitched": product_pitched, "PreferredPropertyStar": preferred_star,
        "MaritalStatus": marital_status, "NumberOfTrips": num_trips,
        "Passport": 1 if passport == "Yes" else 0, "PitchSatisfactionScore": pitch_score,
        "OwnCar": 1 if own_car == "Yes" else 0, "NumberOfChildrenVisiting": num_children,
        "Designation": designation, "MonthlyIncome": monthly_income,
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"✅ Likely to purchase (probability: {probability:.1%})")
    else:
        st.warning(f"❌ Unlikely to purchase (probability: {probability:.1%})")
