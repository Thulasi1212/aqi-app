import streamlit as st
import pandas as pd
import pickle

<<<<<<< HEAD

@st.cache_resource
def load_pipeline():
    with open('aqi_pipeline.pkl','rb') as file:
        pipeline=pickle.load(file)
    return pipeline['trfr'],pipeline['transformer']

trfr,transformer=load_pipeline()

st.title('Air Quality Index Predictor')
st.write('Enter Environmental parameters to predict AQI')

st.sidebar.header('Location')
city=st.sidebar.selectbox('City',[
=======
st.set_page_config(page_title='AQI App', page_icon='🌫️', layout='wide')

# ─────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────
@st.cache_resource
def load_regressor():
    with open('aqi_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline['trfr'], pipeline['transformer']

classifier_available = False
try:
    @st.cache_resource
    def load_classifier():
        with open('aqi_classifier.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['pipeline'], data['label_encoder'], data['classes']
    clf_pipeline, le, classes = load_classifier()
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
>>>>>>> 54187dd (Added AQI classifier)
    'Ahmedabad', 'Aizawl', 'Amaravati', 'Amritsar', 'Bengaluru',
    'Bhopal', 'Brajrajnagar', 'Chandigarh', 'Chennai', 'Coimbatore',
    'Delhi', 'Ernakulam', 'Gurugram', 'Guwahati', 'Hyderabad',
    'Jaipur', 'Jorapokhar', 'Kochi', 'Kolkata', 'Lucknow',
    'Mumbai', 'Patna', 'Shillong', 'Talcher', 'Thiruvananthapuram',
<<<<<<< HEAD
    'Visakhapatnam'])

st.sidebar.header('Pollutant Levels')
pm25 = st.sidebar.number_input('PM2.5 (μg/m³)', 0.0, 500.0, 50.0)
no = st.sidebar.number_input('NO (μg/m³)', 0.0, 500.0, 20.0)
no2 = st.sidebar.number_input('NO2 (μg/m³)', 0.0, 200.0, 40.0)
nox = st.sidebar.number_input('NOx (ppb)', 0.0, 500.0, 50.0)
co = st.sidebar.number_input('CO (mg/m³)', 0.0, 50.0, 1.0)
so2 = st.sidebar.number_input('SO2 (μg/m³)', 0.0, 100.0, 10.0)
o3 = st.sidebar.number_input('O3 (μg/m³)', 0.0, 300.0, 50.0)
benzene = st.sidebar.number_input('Benzene (μg/m³)', 0.0, 50.0, 2.0)

if st.button('Predict AQI',type='primary'):
    input_data=pd.DataFrame({'City': [city],'PM2.5': [pm25],'NO': [no],'NO2': [no2],'NOx': [nox],'CO': [co],'SO2': [so2],'O3': [o3],'Benzene': [benzene]})
    input_processed=transformer.transform(input_data)
    prediction=trfr.predict(input_processed)
    st.success(f'### Predicted AQI:,{prediction[0]:.1f}')
    aqi_value=prediction[0]
    if aqi_value<=50:
        st.info('**Good** - Air qualityis satisfactory')
    elif aqi_value<=100:
        st.warning('**Moderate** - Acceptable air quality')
    elif aqi_value<=150:
        st.warning('**Unhealthy for Sensitive Groups**')
    elif aqi_value<=150:
        st.error('**Unhealthy**')
    else:
        st.error('**Hazardous**')
    with st.expander('View Input Details'):
        st.dataframe(input_data)
=======
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
    st.warning('⚠️ Classifier model (aqi_classifier.pkl) not found. Only regression results will be shown.')

st.divider()

if st.button('🔍 Predict AQI', type='primary', use_container_width=True):

    # ── Build input dataframes ──
    base_input = pd.DataFrame({
        'City': [city], 'PM2.5': [pm25], 'NO': [no], 'NO2': [no2],
        'NOx': [nox], 'CO': [co], 'SO2': [so2], 'O3': [o3], 'Benzene': [benzene]
    })

    clf_input = base_input.copy()
    clf_input['Pollution_Index'] = clf_input['PM2.5'] + clf_input['NO2'] + clf_input['SO2']
    clf_input['NOx_ratio']       = clf_input['NOx'] / (clf_input['NO'] + 1)

    # ── Regression ──
    reg_processed = transformer.transform(base_input)
    aqi_value     = trfr.predict(reg_processed)[0]

    # ── Classification ──
    if classifier_available:
        pred_encoded = clf_pipeline.predict(clf_input)
        pred_proba   = clf_pipeline.predict_proba(clf_input)[0]
        pred_label   = le.inverse_transform(pred_encoded)[0]
        emoji, alert_type, description = category_config.get(pred_label, ('🔵', 'info', ''))

    # ── RESULTS LAYOUT ──
    st.subheader('📊 Results')
    col1, col2 = st.columns(2)

    # Left column — Regression
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

    # Right column — Classification
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
                'Category':    le.classes_,
                'Probability': pred_proba
            }).sort_values('Probability', ascending=False)
            st.bar_chart(proba_df.set_index('Category')['Probability'])
        else:
            st.info('Classifier model not available.')

    # ── Input Summary ──
    st.divider()
    with st.expander('🔎 View Input Details'):
        st.dataframe(base_input, use_container_width=True)
>>>>>>> 54187dd (Added AQI classifier)
