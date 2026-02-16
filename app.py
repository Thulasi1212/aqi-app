import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title='AQI App', page_icon='🌫️', layout='wide')

# ─────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────
@st.cache_resource
def load_regressor():
    with open('aqi_pipeline.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['trfr'], data['transformer']

trfr, transformer = load_regressor()

classifier_available = False
try:
    @st.cache_resource
    def load_classifier():
        with open('aqi_classifier.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['trfr'], data['transformer'], data['label_encoder'], data['classes']
    clf_model, clf_transformer, le, classes = load_classifier()
    classifier_available = True
except Exception:
    pass

# ─────────────────────────────────────────
# CATEGORY CONFIG
# ─────────────────────────────────────────
category_config = {
    'Good':         ('🟢', 'success', 'Air quality is excellent.'),
    'Satisfactory': ('🟡', 'success', 'Acceptable air quality.'),
    'Moderate':     ('🟠', 'warning', 'May cause discomfort to sensitive people.'),
    'Poor':         ('🔴', 'warning', 'Breathing discomfort for most people.'),
    'Very Poor':    ('🟣', 'error',   'Serious health effects for most people.'),
    'Severe':       ('⚫', 'error',   'Hazardous — affects healthy people too.'),
}

# ─────────────────────────────────────────
# SIDEBAR — INPUTS
# ─────────────────────────────────────────
st.sidebar.header('📍 Location')
city = st.sidebar.selectbox('City', [
    'Ahmedabad', 'Aizawl', 'Amaravati', 'Amritsar', 'Bengaluru',
    'Bhopal', 'Brajrajnagar', 'Chandigarh', 'Chennai', 'Coimbatore',
    'Delhi', 'Ernakulam', 'Gurugram', 'Guwahati', 'Hyderabad',
    'Jaipur', 'Jorapokhar', 'Kochi', 'Kolkata', 'Lucknow',
    'Mumbai', 'Patna', 'Shillong', 'Talcher', 'Thiruvananthapuram',
    'Visakhapatnam'
])

st.sidebar.header('💨 Pollutant Levels')
pm25    = st.sidebar.number_input('PM2.5 (μg/m³)', 0.0, 500.0, 50.0)
no      = st.sidebar.number_input('NO (μg/m³)',     0.0, 500.0, 20.0)
no2     = st.sidebar.number_input('NO2 (μg/m³)',    0.0, 200.0, 40.0)
nox     = st.sidebar.number_input('NOx (ppb)',      0.0, 500.0, 50.0)
co      = st.sidebar.number_input('CO (mg/m³)',     0.0,  50.0,  1.0)
so2     = st.sidebar.number_input('SO2 (μg/m³)',    0.0, 100.0, 10.0)
o3      = st.sidebar.number_input('O3 (μg/m³)',     0.0, 300.0, 50.0)
benzene = st.sidebar.number_input('Benzene (μg/m³)',0.0,  50.0,  2.0)

# ─────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────
st.title('🌫️ Air Quality Index App')
st.write('Enter pollutant levels in the sidebar and click **Predict** to get both AQI score and category.')

if not classifier_available:
    st.warning('Classifier model not found. Only regression results will be shown.')

st.divider()

if st.button('🔍 Predict AQI', type='primary', use_container_width=True):

    # Build full input with engineered features (both models use same 11 columns)
    input_df = pd.DataFrame({
        'City': [city], 'PM2.5': [pm25], 'NO': [no], 'NO2': [no2],
        'NOx': [nox], 'CO': [co], 'SO2': [so2], 'O3': [o3], 'Benzene': [benzene],
        'Pollution_Index': [pm25 + no2 + so2],
        'NOx_ratio':       [nox / (no + 1)]
    })

    # Regression prediction
    reg_processed = transformer.transform(input_df)
    aqi_value     = trfr.predict(reg_processed)[0]

    # Classification prediction
    if classifier_available:
        clf_processed = clf_transformer.transform(input_df)
        pred_encoded  = clf_model.predict(clf_processed)
        pred_proba    = clf_model.predict_proba(clf_processed)[0]
        pred_label    = le.inverse_transform(pred_encoded)[0]
        emoji, alert_type, description = category_config.get(pred_label, ('🔵', 'info', ''))

    # Results Layout
    st.subheader('📊 Results')
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('### 📈 AQI Score')
        st.metric(label='Predicted AQI', value=f'{aqi_value:.1f}')

        if aqi_value <= 50:
            st.success('🟢 **Good** — Air quality is excellent')
        elif aqi_value <= 100:
            st.success('🟡 **Satisfactory** — Acceptable air quality')
        elif aqi_value <= 200:
            st.warning('🟠 **Moderate** — May cause discomfort to sensitive people')
        elif aqi_value <= 300:
            st.warning('🔴 **Poor** — Breathing discomfort for most people')
        elif aqi_value <= 400:
            st.error('🟣 **Very Poor** — Serious health effects likely')
        else:
            st.error('⚫ **Severe** — Hazardous for everyone')

    with col2:
        st.markdown('### 🏷️ AQI Category')
        if classifier_available:
            st.metric(label='Predicted Category', value=f'{emoji} {pred_label}')

            if alert_type == 'success':
                st.success(f'_{description}_')
            elif alert_type == 'warning':
                st.warning(f'_{description}_')
            else:
                st.error(f'_{description}_')

            st.markdown('**Confidence per Category:**')
            proba_df = pd.DataFrame({
                'Category':    classes,
                'Probability': pred_proba
            }).sort_values('Probability', ascending=False)
            st.bar_chart(proba_df.set_index('Category')['Probability'])
        else:
            st.info('Classifier model not available.')

    st.divider()
    with st.expander('🔎 View Input Details'):
        st.dataframe(
            input_df[['City','PM2.5','NO','NO2','NOx','CO','SO2','O3','Benzene']],
            use_container_width=True
        )
