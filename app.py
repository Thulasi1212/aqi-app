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

/* Hide Streamlit top toolbar (share/edit/github bar) */
header[data-testid="stHeader"] { display: none !important; }
#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
.viewerBadge_container__1QSob { display: none !important; }

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
    reg_transformer = reg_loaded['transformer']    # ColumnTransformer (not a pipeline)
    reg_model       = reg_loaded['trfr']            # RandomForestRegressor
    clf_pipeline    = clf_loaded['pipeline']        # Full Pipeline
    clf_le          = clf_loaded['label_encoder']   # LabelEncoder
    # Hardcoded from pkl inspection: {0:'Good',1:'Moderate',2:'Poor',3:'Satisfactory',4:'Severe',5:'Very Poor'}
    # Also try to read from classes_ dynamically in case pkl changes
    try:
        classes = [str(c) for c in clf_le.classes_]
        # classes_ is already strings if training was done correctly
        if all(not c.isdigit() for c in classes):
            label_mapping = {i: c for i, c in enumerate(classes)}
        else:
            raise ValueError("classes_ are integers")
    except Exception:
        label_mapping = {0:'Good', 1:'Moderate', 2:'Poor', 3:'Satisfactory', 4:'Severe', 5:'Very Poor'}
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

    predict_clicked = st.button("🔍 Predict & Classify AQI")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    city = st.selectbox("📍 City", CITIES)
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

    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════
# RIGHT: RESULTS
# ═══════════════════════════════
with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    # ── Panel header + legend on same row ──
    _chips = ""
    for _idx in sorted(label_mapping.keys()):
        _cat = label_mapping[_idx]
        _dot = AQI_COLORS.get(_cat, "#4ecaa0")
        _chips += (
            f'<span style="display:inline-flex;align-items:center;gap:3px;margin-left:4px;'
            f'background:rgba(255,255,255,0.04);border:1px solid rgba(78,202,160,0.1);'
            f'border-radius:4px;padding:1px 5px;white-space:nowrap;">'
            f'<span style="width:6px;height:6px;border-radius:50%;background:{_dot};display:inline-block;"></span>'
            f'<span style="color:#c0ccc8;font-size:0.65rem;">{_idx}=<b style="color:#e8f4f0;">{_cat}</b></span>'
            f'</span>'
        )
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;">'
        f'<span style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:#e8f4f0;">📊 Prediction Results</span>'
        f'<div style="display:flex;flex-wrap:wrap;justify-content:flex-end;align-items:center;">'
        f'<span style="color:#4a7a68;font-size:0.62rem;margin-right:3px;">🔑</span>{_chips}'
        f'</div></div>',
        unsafe_allow_html=True
    )

    # Run prediction when button clicked
    if predict_clicked:
        input_df = pd.DataFrame([{
            'City': city, 'PM2.5': pm25, 'NO': no, 'NO2': no2,
            'NOx': nox, 'CO': co, 'SO2': so2, 'O3': o3, 'Benzene': benzene
        }])
        try:
            # Regressor (manual transform — not a pipeline)
            X_trans   = reg_transformer.transform(input_df)
            aqi_value = float(reg_model.predict(X_trans)[0])

            # Classifier — pipeline handles transform internally
            aqi_encoded  = int(clf_pipeline.predict(input_df)[0])
            aqi_category = label_mapping[aqi_encoded]  # e.g. 2 -> 'Poor'

            # Build prob_data NOW with plain Python strings — no numpy in session_state
            proba_raw = clf_pipeline.predict_proba(input_df)[0]
            sorted_keys = sorted(label_mapping.keys())
            prob_categories = [str(label_mapping[i]) for i in sorted_keys]
            prob_values     = [round(float(proba_raw[i]) * 100, 2) for i in sorted_keys]

            st.session_state.results = {
                'aqi_value':       aqi_value,
                'aqi_category':    aqi_category,
                'prob_categories': prob_categories,  # ['Good','Moderate','Poor',...]
                'prob_values':     prob_values,       # [5.2, 12.3, 45.1, ...]
                'city':            city,
                'input_df':        input_df.to_dict('records'),
            }
        except Exception as e:
            st.error(f'Prediction failed: {e}')
            st.session_state.results = None

    # Render results (persists across reruns via session_state)
    if st.session_state.results:
        r = st.session_state.results
        color = AQI_COLORS.get(r['aqi_category'], '#4ecaa0')

        # ── Top cards: AQI value + Category ──
        card1, card2 = st.columns(2)

        with card1:
            st.markdown(f'''
            <div class="result-card">
                <div class="section-label">📈 Regression · AQI Value</div>
                <div class="aqi-number">{r["aqi_value"]:.1f}</div>
                <div class="model-tag">RandomForest Regressor</div>
            </div>
            ''', unsafe_allow_html=True)

        with card2:
            # Ensure string label — if aqi_category is still an int, decode it
            _display_cat = r["aqi_category"]
            if str(_display_cat).isdigit():
                _display_cat = label_mapping.get(int(_display_cat), _display_cat)
            _cat_color = AQI_COLORS.get(str(_display_cat), '#4ecaa0')
            st.markdown(f'''
            <div class="result-card">
                <div class="section-label">🏷️ Classification · Category</div>
                <div class="aqi-category" style="color:{_cat_color}; margin: 0.5rem 0;">● {_display_cat}</div>
                <div class="model-tag">XGBoost Classifier</div>
            </div>
            ''', unsafe_allow_html=True)

        # ── Advisory ──
        st.info(ADVISORIES.get(r['aqi_category'], 'Monitor air quality regularly.'))

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ── Classifier Report ──
        if r['prob_categories']:
            st.markdown('**🧾 Classifier Report**')

            CAT_COLORS = {
                "Good":                "#00c853",
                "Satisfactory":        "#aeea00",
                "Moderate":            "#ffd600",
                "Moderately Polluted": "#ffd600",
                "Poor":                "#ff6d00",
                "Very Poor":           "#dd2c00",
                "Severe":              "#c0392b",
            }



            # HTML/CSS horizontal bar chart — zero dependencies
            pairs = sorted(zip(r['prob_categories'], r['prob_values']), key=lambda x: -x[1])
            bars = ""
            for cat, prob in pairs:
                col   = CAT_COLORS.get(cat, "#4ecaa0")
                pred  = cat == r['aqi_category']
                style = "font-weight:700;color:#4ecaa0;" if pred else "color:#b0c4be;"
                star  = " ★" if pred else ""
                bars += (
                    f'<div style="margin-bottom:9px;">' 
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:2px;">' 
                    f'<span style="font-size:0.8rem;{style}">{cat}{star}</span>' 
                    f'<span style="font-size:0.8rem;{style}">{prob:.1f}%</span>' 
                    f'</div>' 
                    f'<div style="background:rgba(255,255,255,0.06);border-radius:5px;height:9px;">' 
                    f'<div style="width:{min(prob,100)}%;height:100%;background:{col};border-radius:5px;"></div>' 
                    f'</div></div>'
                )
            st.markdown(
                f'<div style="padding:0.3rem 0;">{bars}' 
                f'<div style="color:#3d6b5a;font-size:0.7rem;margin-top:6px;">★ = predicted category</div></div>',
                unsafe_allow_html=True
            )

        # ── Input summary ──
        with st.expander("📋 View Input Summary"):
            st.dataframe(pd.DataFrame(r['input_df']), use_container_width=True, hide_index=True)

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
