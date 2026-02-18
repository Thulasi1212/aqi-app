import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="AQI Predictor", page_icon="🌫️", layout="centered")
st.title("🌫️ AQI Predictor")
st.markdown("Enter pollutant values and city to predict **AQI** and **AQI Bucket**.")

# ─── Load both pkl files ───────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("aqi_pipeline.pkl", "rb") as f:
        reg_bundle = pickle.load(f)
    # reg_bundle["transformer"] → ColumnTransformer (StandardScaler + OrdinalEncoder)
    # reg_bundle["trfr"]        → RandomForestRegressor

    with open("aqi_classifier.pkl", "rb") as f:
        clf_bundle = pickle.load(f)
    # clf_bundle["clf_transformer"] → ColumnTransformer (RobustScaler + OrdinalEncoder)
    # clf_bundle["xgb"]            → XGBClassifier (standalone, NOT a Pipeline)
    # clf_bundle["label_encoder"]  → LabelEncoder
    # clf_bundle["classes"]        → le.classes_.tolist()

    return reg_bundle, clf_bundle

try:
    reg_bundle, clf_bundle = load_models()
    models_loaded = True
except FileNotFoundError as e:
    st.error(f"❌ Model file not found: {e}. "
             "Place aqi_pipeline.pkl and aqi_classifier.pkl in the same folder as app.py.")
    models_loaded = False

# ─── Input form ───────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📥 Input Features")

cities = sorted([
    "Ahmedabad", "Aizawl", "Amaravati", "Amritsar", "Bengaluru", "Bhopal",
    "Brajrajnagar", "Chandigarh", "Chennai", "Coimbatore", "Delhi", "Ernakulam",
    "Gurugram", "Guwahati", "Hyderabad", "Jaipur", "Jorapokhar", "Kochi",
    "Kolkata", "Lucknow", "Mumbai", "Munger", "Nagpur", "Patna", "Shillong",
    "Talcher", "Thiruvananthapuram", "Visakhapatnam"
])

col1, col2 = st.columns(2)

with col1:
    city    = st.selectbox("🏙️ City", cities)
    pm25    = st.number_input("PM2.5 (µg/m³)",  min_value=0.0, max_value=1000.0, value=50.0,  step=0.1)
    no      = st.number_input("NO (µg/m³)",      min_value=0.0, max_value=500.0,  value=10.0,  step=0.1)
    no2     = st.number_input("NO2 (µg/m³)",     min_value=0.0, max_value=500.0,  value=20.0,  step=0.1)
    nox     = st.number_input("NOx (ppb)",        min_value=0.0, max_value=500.0,  value=30.0,  step=0.1)

with col2:
    co      = st.number_input("CO (mg/m³)",       min_value=0.0, max_value=100.0,  value=1.0,   step=0.01)
    so2     = st.number_input("SO2 (µg/m³)",      min_value=0.0, max_value=500.0,  value=10.0,  step=0.1)
    o3      = st.number_input("O3 (µg/m³)",       min_value=0.0, max_value=500.0,  value=30.0,  step=0.1)
    benzene = st.number_input("Benzene (µg/m³)",  min_value=0.0, max_value=100.0,  value=1.0,   step=0.01)

# ─── Predict button ────────────────────────────────────────────────────────────
st.markdown("---")

if st.button("🔍 Predict AQI & AQI Bucket", use_container_width=True, disabled=not models_loaded):

    input_df = pd.DataFrame([{
        "PM2.5":   pm25,
        "NO":      no,
        "NO2":     no2,
        "NOx":     nox,
        "CO":      co,
        "SO2":     so2,
        "O3":      o3,
        "Benzene": benzene,
        "City":    city
    }])

    # ── REGRESSION (AQI value) ──────────────────────────────────────────────
    # Both saved separately → manual 2-step: transform then predict
    reg_transformer = reg_bundle["transformer"]   # ColumnTransformer
    rf_model        = reg_bundle["trfr"]           # RandomForestRegressor
    x_reg_trans     = reg_transformer.transform(input_df)
    aqi_predicted   = rf_model.predict(x_reg_trans)[0]

    # ── CLASSIFICATION (AQI Bucket) ─────────────────────────────────────────
    # Both saved separately → manual 2-step: transform then predict
    clf_transformer = clf_bundle["clf_transformer"]  # ColumnTransformer
    xgb_model       = clf_bundle["xgb"]              # XGBClassifier
    label_encoder   = clf_bundle["label_encoder"]    # LabelEncoder
    x_clf_trans     = clf_transformer.transform(input_df)
    bucket_encoded  = xgb_model.predict(x_clf_trans)[0]
    bucket_label    = label_encoder.inverse_transform([bucket_encoded])[0]

    # ── Display results ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Prediction Results")

    res1, res2 = st.columns(2)
    with res1:
        st.metric(label="🌡️ Predicted AQI", value=f"{aqi_predicted:.2f}")
    with res2:
        bucket_icons = {
            "Good":                "🟢",
            "Satisfactory":        "🟡",
            "Moderately Polluted": "🟠",
            "Poor":                "🔴",
            "Very Poor":           "🟣",
            "Severe":              "⚫",
        }
        icon = bucket_icons.get(bucket_label, "🔵")
        st.metric(label="🗂️ AQI Bucket", value=f"{icon} {bucket_label}")

    # ── Health advisory ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💡 Health Advisory")
    if aqi_predicted <= 50:
        st.success("**Good** — Air quality is satisfactory. Enjoy outdoor activities.")
    elif aqi_predicted <= 100:
        st.info("**Satisfactory** — Acceptable air quality. Sensitive people should limit prolonged outdoor exertion.")
    elif aqi_predicted <= 200:
        st.warning("**Moderately Polluted** — Sensitive groups may experience health effects.")
    elif aqi_predicted <= 300:
        st.warning("**Poor** — Everyone may begin to experience health effects. Sensitive groups should limit outdoor activity.")
    elif aqi_predicted <= 400:
        st.error("**Very Poor** — Health alert: everyone may experience serious health effects. Avoid outdoor activity.")
    else:
        st.error("**Severe** — Health emergency. Stay indoors and keep windows closed.")