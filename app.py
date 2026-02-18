import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌿",
    layout="wide"          # Wide layout = side-by-side columns
)

# ── CUSTOM CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3, h4 { font-family: 'Syne', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f1923 0%, #1a2d3d 50%, #0f2318 100%);
    color: #e8f4f0;
    min-height: 100vh;
}

.block-container { padding: 1.5rem 2rem 1rem 2rem; }

.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #4ecaa0, #56d4e8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
}

.subtitle {
    color: #7fb8a8;
    font-size: 0.88rem;
    margin-bottom: 1.2rem;
}

.section-label {
    color: #7fb8a8;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}

.result-card {
    background: rgba(78, 202, 160, 0.07);
    border: 1px solid rgba(78, 202, 160, 0.25);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.9rem;
    text-align: center;
}

.aqi-number {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    color: #4ecaa0;
    line-height: 1.1;
}

.aqi-category {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    line-height: 1.2;
}

.model-tag {
    color: #4a7a68;
    font-size: 0.75rem;
    margin-top: 0.4rem;
}

.panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(78,202,160,0.12);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    height: 100%;
}

.divider {
    border: none;
    border-top: 1px solid rgba(78,202,160,0.12);
    margin: 1rem 0;
}

/* Inputs */
div[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(78,202,160,0.2) !important;
    border-radius: 7px !important;
    color: #e8f4f0 !important;
    font-size: 0.88rem !important;
}

div[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(78,202,160,0.2) !important;
    border-radius: 7px !important;
    color: #e8f4f0 !important;
}

/* Button */
div[data-testid="stButton"] > button {
    background: linear-gradient(90deg, #4ecaa0, #38b48a);
    color: #0f1923;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.5rem;
    width: 100%;
    margin-top: 0.5rem;
}
div[data-testid="stButton"] > button:hover { opacity: 0.85; }

/* Dataframe */
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* Placeholder text when no results yet */
.placeholder {
    color: #3d6b5a;
    font-size: 0.9rem;
    text-align: center;
    padding: 3rem 1rem;
    border: 1px dashed rgba(78,202,160,0.15);
    border-radius: 14px;
    margin-top: 1rem;
}
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
    reg_transformer = reg_loaded['transformer']   # ColumnTransformer (not a pipeline)
    reg_model       = reg_loaded['trfr']           # RandomForestRegressor
    clf_pipeline    = clf_loaded['pipeline']       # Full Pipeline
    clf_le          = clf_loaded['label_encoder']  # LabelEncoder
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ── CONSTANTS ─────────────────────────────────────────────
AQI_COLORS = {
    "Good":                "#00c853",
    "Satisfactory":        "#aeea00",
    "Moderately Polluted": "#ffd600",
    "Poor":                "#ff6d00",
    "Very Poor":           "#dd2c00",
    "Severe":              "#c0392b",
}

ADVISORIES = {
    "Good":                "✅ Air quality is satisfactory. Enjoy outdoor activities freely.",
    "Satisfactory":        "🟡 Acceptable air quality. Sensitive individuals should limit prolonged exertion outdoors.",
    "Moderately Polluted": "⚠️ Sensitive groups may experience health effects. General public less likely affected.",
    "Poor":                "🟠 Everyone may begin to experience health effects. Sensitive groups should avoid outdoor activity.",
    "Very Poor":           "🔴 Health alert — everyone may experience serious effects. Avoid all outdoor activities.",
    "Severe":              "🚨 Emergency conditions. Entire population likely affected. Stay indoors.",
}

# ── CITY LIST ─────────────────────────────────────────────
CITIES = []
for name, t, cols in reg_transformer.transformers:
    if name == 'cat' and hasattr(t, 'categories_'):
        CITIES = sorted(t.categories_[0].tolist())
        break
if not CITIES:
    CITIES = sorted([
        'Ahmedabad','Aizawl','Amaravati','Amritsar','Bengaluru','Bhopal',
        'Brajrajnagar','Chandigarh','Chennai','Coimbatore','Delhi','Ernakulam',
        'Gurugram','Guwahati','Hyderabad','Jaipur','Jorapokhar','Kochi',
        'Kolkata','Lucknow','Mumbai','Nagpur','Patna','Shillong','Talcher',
        'Thiruvananthapuram','Visakhapatnam'
    ])

# ── SESSION STATE ─────────────────────────────────────────
if 'results' not in st.session_state:
    st.session_state.results = None

# ── HEADER ────────────────────────────────────────────────
st.markdown('<div class="main-title">🌿 AQI Predictor & Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter pollutant readings to predict the AQI value and classify the air quality category</div>', unsafe_allow_html=True)
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── MAIN TWO-COLUMN LAYOUT ────────────────────────────────
left_col, right_col = st.columns([1, 1.4], gap="large")

# ═══════════════════════════════
# LEFT: INPUTS
# ═══════════════════════════════
with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### 📋 Input Parameters")

    city = st.selectbox("📍 City", CITIES)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("**🧪 Pollutant Readings**")

    c1, c2 = st.columns(2)
    with c1:
        pm25    = st.number_input("PM2.5 (µg/m³)",  min_value=0.0, value=45.0,  step=0.1)
        no      = st.number_input("NO (µg/m³)",      min_value=0.0, value=10.0,  step=0.1)
        no2     = st.number_input("NO2 (µg/m³)",     min_value=0.0, value=25.0,  step=0.1)
        nox     = st.number_input("NOx (µg/m³)",     min_value=0.0, value=35.0,  step=0.1)
    with c2:
        co      = st.number_input("CO (mg/m³)",      min_value=0.0, value=1.2,   step=0.01)
        so2     = st.number_input("SO2 (µg/m³)",     min_value=0.0, value=15.0,  step=0.1)
        o3      = st.number_input("O3 (µg/m³)",      min_value=0.0, value=40.0,  step=0.1)
        benzene = st.number_input("Benzene (µg/m³)", min_value=0.0, value=2.5,   step=0.01)

    predict_clicked = st.button("🔍 Predict & Classify AQI")
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════
# RIGHT: RESULTS
# ═══════════════════════════════
with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### 📊 Prediction Results")

    # Run prediction when button clicked
    if predict_clicked:
        input_df = pd.DataFrame([{
            'City': city, 'PM2.5': pm25, 'NO': no, 'NO2': no2,
            'NOx': nox, 'CO': co, 'SO2': so2, 'O3': o3, 'Benzene': benzene
        }])
        try:
            # Regressor (manual transform — not a pipeline)
            X_trans   = reg_transformer.transform(input_df)
            aqi_value = reg_model.predict(X_trans)[0]

            # Classifier (pipeline handles transform internally)
            aqi_encoded  = clf_pipeline.predict(input_df)[0]
            # ✅ inverse_transform: converts numeric label (0,1,2...) → 'Good','Poor' etc.
            aqi_category = clf_le.inverse_transform([aqi_encoded])[0]

            # Probabilities — inverse_transform class names for all classes
            proba = clf_pipeline.predict_proba(input_df)[0] if hasattr(clf_pipeline, 'predict_proba') else None
            # clf_le.classes_ already holds the string labels in encoded order
            class_labels = clf_le.classes_   # e.g. ['Good','Moderately Polluted','Poor',...]

            st.session_state.results = {
                'aqi_value':    aqi_value,
                'aqi_category': aqi_category,
                'proba':        proba,
                'class_labels': class_labels,
                'city':         city,
                'input_df':     input_df,
            }
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.session_state.results = None

    # Render results (persists across reruns via session_state)
    if st.session_state.results:
        r = st.session_state.results
        color = AQI_COLORS.get(r['aqi_category'], "#4ecaa0")

        # ── Top cards: AQI value + Category ──
        card1, card2 = st.columns(2)

        with card1:
            st.markdown(f"""
            <div class="result-card">
                <div class="section-label">📈 Regression · AQI Value</div>
                <div class="aqi-number">{r['aqi_value']:.1f}</div>
                <div class="model-tag">RandomForest Regressor</div>
            </div>
            """, unsafe_allow_html=True)

        with card2:
            st.markdown(f"""
            <div class="result-card">
                <div class="section-label">🏷️ Classification · Category</div>
                <div class="aqi-category" style="color:{color}; margin: 0.5rem 0;">● {r['aqi_category']}</div>
                <div class="model-tag">XGBoost Classifier</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Advisory ──
        st.info(ADVISORIES.get(r['aqi_category'], "Monitor air quality regularly."))

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ── Classifier Report ──
        if r['proba'] is not None:
            st.markdown("**🧾 Classifier Report — Probability per Category**")

            # inverse_transform every class index → guaranteed string labels
            # clf_le.classes_ order matches proba array order exactly
            n_classes     = len(r['proba'])
            all_indices   = list(range(n_classes))
            # inverse_transform converts [0,1,2,...] → ['Good','Poor','Severe',...]
            decoded_names = [str(clf_le.inverse_transform([i])[0]) for i in all_indices]

            prob_df = pd.DataFrame({
                'Category':      decoded_names,
                'Probability %': [round(float(p) * 100, 2) for p in r['proba']]
            }).sort_values('Probability %', ascending=False).reset_index(drop=True)

            def highlight_row(row):
                if row['Category'] == r['aqi_category']:
                    return ['background-color: rgba(78,202,160,0.18); font-weight:bold; color:#4ecaa0'] * len(row)
                return ['color:#c0ccc8'] * len(row)

            styled = (
                prob_df.style
                .apply(highlight_row, axis=1)
                .format({'Probability %': '{:.2f}%'})
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # Bar chart using plotly — ensures string category labels on x-axis
            import plotly.graph_objects as go

            bar_colors = [
                AQI_COLORS.get(cat, "#4ecaa0") if cat == r['aqi_category'] else "rgba(78,202,160,0.35)"
                for cat in prob_df['Category']
            ]

            fig = go.Figure(go.Bar(
                x=prob_df['Category'].tolist(),        # string labels from inverse_transform
                y=prob_df['Probability %'].tolist(),
                marker_color=bar_colors,
                text=[f"{v:.1f}%" for v in prob_df['Probability %']],
                textposition='outside',
                textfont=dict(color='#e8f4f0', size=11),
            ))
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e8f4f0', family='DM Sans'),
                xaxis=dict(
                    tickfont=dict(size=11, color='#7fb8a8'),
                    gridcolor='rgba(78,202,160,0.08)',
                    title=None,
                ),
                yaxis=dict(
                    tickfont=dict(size=11, color='#7fb8a8'),
                    gridcolor='rgba(78,202,160,0.08)',
                    title='Probability %',
                    titlefont=dict(color='#7fb8a8'),
                ),
                margin=dict(t=20, b=10, l=10, r=10),
                height=280,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Input summary ──
        with st.expander("📋 View Input Summary"):
            st.dataframe(r['input_df'], use_container_width=True, hide_index=True)

    else:
        st.markdown("""
        <div class="placeholder">
            🌿 Enter pollutant values on the left<br>and click <b>Predict & Classify AQI</b><br>to see results here.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; color:#3d6b5a; font-size:0.78rem;'>AQI Predictor · RandomForest Regressor + XGBoost Classifier</div>",
    unsafe_allow_html=True
)
