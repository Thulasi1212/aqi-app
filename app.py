import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌿",
    layout="centered"
)

# ── CUSTOM CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f1923 0%, #1a2d3d 50%, #0f2318 100%);
    color: #e8f4f0;
}

.block-container { padding-top: 2rem; }

.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #4ecaa0, #56d4e8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.2rem;
}

.subtitle {
    text-align: center;
    color: #7fb8a8;
    font-size: 0.95rem;
    margin-bottom: 2rem;
}

.result-box {
    background: rgba(78, 202, 160, 0.08);
    border: 1px solid rgba(78, 202, 160, 0.3);
    border-radius: 16px;
    padding: 1.8rem;
    margin-top: 1.5rem;
    text-align: center;
}

.aqi-value {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    color: #4ecaa0;
}

.aqi-label {
    font-size: 1.3rem;
    font-weight: 600;
    margin-top: 0.3rem;
}

.divider {
    border: none;
    border-top: 1px solid rgba(78,202,160,0.15);
    margin: 1.5rem 0;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(78,202,160,0.2) !important;
    border-radius: 8px !important;
    color: #e8f4f0 !important;
}

div[data-testid="stButton"] > button {
    background: linear-gradient(90deg, #4ecaa0, #38b48a);
    color: #0f1923;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2rem;
    width: 100%;
    transition: opacity 0.2s;
}
div[data-testid="stButton"] > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODELS ───────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('aqi_predictor.pkl', 'rb') as f:
        reg = pickle.load(f)
    with open('aqi_classifier.pkl', 'rb') as f:
        clf = pickle.load(f)
    return reg, clf

try:
    reg_loaded, clf_loaded = load_models()
    # Regressor: NOT a pipeline — separate transformer and model
    reg_transformer = reg_loaded['transformer']   # ColumnTransformer
    reg_model       = reg_loaded['trfr']           # RandomForestRegressor
    # Classifier: full Pipeline + LabelEncoder
    clf_pipeline    = clf_loaded['pipeline']       # Pipeline(preprocessor + model)
    clf_le          = clf_loaded['label_encoder']  # LabelEncoder
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ── AQI CATEGORY COLORS ───────────────────────────────────
AQI_COLORS = {
    "Good":                "#00e400",
    "Satisfactory":        "#92d050",
    "Moderately Polluted": "#ffff00",
    "Poor":                "#ff7e00",
    "Very Poor":           "#ff0000",
    "Severe":              "#c0392b",
}

# ── CITY LIST ─────────────────────────────────────────────
# Try to extract from OrdinalEncoder categories in reg_transformer
CITIES = []
for name, t, cols in reg_transformer.transformers:
    if name == 'cat' and hasattr(t, 'categories_'):
        CITIES = sorted(t.categories_[0].tolist())
        break

if not CITIES:
    # Fallback city list (standard Indian AQI dataset cities)
    CITIES = sorted([
        'Ahmedabad', 'Aizawl', 'Amaravati', 'Amritsar', 'Bengaluru',
        'Bhopal', 'Brajrajnagar', 'Chandigarh', 'Chennai', 'Coimbatore',
        'Delhi', 'Ernakulam', 'Gurugram', 'Guwahati', 'Hyderabad',
        'Jaipur', 'Jorapokhar', 'Kochi', 'Kolkata', 'Lucknow',
        'Mumbai', 'Nagpur', 'Patna', 'Shillong', 'Talcher',
        'Thiruvananthapuram', 'Visakhapatnam'
    ])

# ── UI ────────────────────────────────────────────────────
st.markdown('<div class="main-title">🌿 AQI Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict Air Quality Index value & classify category from pollutant readings</div>', unsafe_allow_html=True)

# ── INPUTS ────────────────────────────────────────────────
st.markdown("### 📍 Location")
city = st.selectbox("City", CITIES, label_visibility="collapsed")

st.markdown("### 🧪 Pollutant Readings")
col1, col2 = st.columns(2)

with col1:
    pm25    = st.number_input("PM2.5 (µg/m³)",  min_value=0.0, value=45.0,  step=0.1)
    no      = st.number_input("NO (µg/m³)",      min_value=0.0, value=10.0,  step=0.1)
    no2     = st.number_input("NO2 (µg/m³)",     min_value=0.0, value=25.0,  step=0.1)
    nox     = st.number_input("NOx (µg/m³)",     min_value=0.0, value=35.0,  step=0.1)

with col2:
    co      = st.number_input("CO (mg/m³)",      min_value=0.0, value=1.2,   step=0.01)
    so2     = st.number_input("SO2 (µg/m³)",     min_value=0.0, value=15.0,  step=0.1)
    o3      = st.number_input("O3 (µg/m³)",      min_value=0.0, value=40.0,  step=0.1)
    benzene = st.number_input("Benzene (µg/m³)", min_value=0.0, value=2.5,   step=0.01)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── BUTTON ────────────────────────────────────────────────
predict_clicked = st.button("🔍 Predict & Classify AQI")

# ── RESULT PLACEHOLDER (renders above inputs on rerun via session_state) ──
result_slot = st.empty()

if predict_clicked:
    input_df = pd.DataFrame([{
        'City':    city,
        'PM2.5':   pm25,
        'NO':      no,
        'NO2':     no2,
        'NOx':     nox,
        'CO':      co,
        'SO2':     so2,
        'O3':      o3,
        'Benzene': benzene
    }])

    try:
        # Regressor: manually transform then predict (not a pipeline)
        X_transformed  = reg_transformer.transform(input_df)
        aqi_value      = reg_model.predict(X_transformed)[0]

        # Classifier: pipeline handles transform + predict internally
        aqi_encoded    = clf_pipeline.predict(input_df)[0]
        aqi_category   = clf_le.inverse_transform([aqi_encoded])[0]

        # Class probabilities for classifier report
        if hasattr(clf_pipeline, 'predict_proba'):
            proba       = clf_pipeline.predict_proba(input_df)[0]
            class_names = clf_le.classes_
        else:
            proba       = None
            class_names = clf_le.classes_

        color = AQI_COLORS.get(aqi_category, "#4ecaa0")

        advisories = {
            "Good":                "✅ Air quality is satisfactory. Enjoy outdoor activities freely.",
            "Satisfactory":        "🟡 Acceptable air quality. Unusually sensitive people should consider limiting prolonged outdoor exertion.",
            "Moderately Polluted": "⚠️ Sensitive groups may experience health effects. General public is less likely to be affected.",
            "Poor":                "🟠 Everyone may begin to experience health effects. Sensitive groups should avoid outdoor activity.",
            "Very Poor":           "🔴 Health alert — everyone may experience serious effects. Avoid all outdoor activities.",
            "Severe":              "🚨 Emergency conditions. The entire population is likely to be affected. Stay indoors.",
        }

        # ── RESULT SECTION ────────────────────────────────
        with result_slot.container():
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("## 📊 Prediction Results")

            # ── ROW 1: AQI Value + Category side by side ──
            r1, r2 = st.columns(2)

            with r1:
                st.markdown(f"""
                <div class="result-box">
                    <div style="color:#7fb8a8; font-size:0.8rem; letter-spacing:0.12em; margin-bottom:0.4rem;">
                        📈 REGRESSION · AQI VALUE
                    </div>
                    <div class="aqi-value">{aqi_value:.1f}</div>
                    <div style="color:#7fb8a8; font-size:0.8rem; margin-top:0.6rem;">
                        RandomForest Regressor
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with r2:
                st.markdown(f"""
                <div class="result-box">
                    <div style="color:#7fb8a8; font-size:0.8rem; letter-spacing:0.12em; margin-bottom:0.4rem;">
                        🏷️ CLASSIFICATION · CATEGORY
                    </div>
                    <div class="aqi-label" style="color:{color}; font-size:1.6rem; font-weight:800; margin: 0.6rem 0;">
                        ● {aqi_category}
                    </div>
                    <div style="color:#7fb8a8; font-size:0.8rem; margin-top:0.6rem;">
                        XGBoost Classifier
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Advisory ──────────────────────────────────
            st.info(advisories.get(aqi_category, "Monitor air quality regularly."))

            # ── CLASSIFIER REPORT: Probability per class ──
            if proba is not None:
                st.markdown("### 🧾 Classifier Report")
                st.markdown("Probability distribution across all AQI categories:")

                prob_df = pd.DataFrame({
                    'Category':    class_names,
                    'Probability': [round(float(p) * 100, 2) for p in proba]
                }).sort_values('Probability', ascending=False).reset_index(drop=True)

                # Highlight predicted class
                def highlight_predicted(row):
                    if row['Category'] == aqi_category:
                        return ['background-color: rgba(78,202,160,0.2); font-weight:bold'] * len(row)
                    return [''] * len(row)

                styled_df = prob_df.style.apply(highlight_predicted, axis=1).format({'Probability': '{:.2f}%'})
                st.dataframe(styled_df, use_container_width=True, hide_index=True)

                # Bar chart
                st.bar_chart(
                    prob_df.set_index('Category')['Probability'],
                    use_container_width=True,
                    color="#4ecaa0"
                )

            # ── Input Summary ─────────────────────────────
            with st.expander("📋 Input Summary"):
                st.dataframe(input_df, use_container_width=True, hide_index=True)

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ── FOOTER ────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; color:#3d6b5a; font-size:0.8rem;'>AQI Predictor · RandomForest Regressor + XGBoost Classifier</div>",
    unsafe_allow_html=True
)
