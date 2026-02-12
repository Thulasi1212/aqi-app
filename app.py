import streamlit as st
import pandas as pd
import pickle


@st.cache_resource
def load_pipeline():
    with open('aqi_pipeline_fresh.pkl','rb') as file:
        pipeline=pickle.load(file)
    return pipeline['trfr'],pipeline['transformer']

trfr,transformer=load_pipeline()

st.title('Air Quality Index Predictor')
st.write('Enter Environmental parameters to predict AQI')

st.sidebar.header('Location')
city=st.sidebar.selectbox('City',[
    'Ahmedabad', 'Aizawl', 'Amaravati', 'Amritsar', 'Bengaluru',
    'Bhopal', 'Brajrajnagar', 'Chandigarh', 'Chennai', 'Coimbatore',
    'Delhi', 'Ernakulam', 'Gurugram', 'Guwahati', 'Hyderabad',
    'Jaipur', 'Jorapokhar', 'Kochi', 'Kolkata', 'Lucknow',
    'Mumbai', 'Patna', 'Shillong', 'Talcher', 'Thiruvananthapuram',
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