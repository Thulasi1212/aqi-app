# 🌿 AQI Predictor & Classifier

A machine learning web app that predicts the **Air Quality Index (AQI) value** and **classifies the air quality category** based on pollutant readings from Indian cities.

Built with XGBoost + Scikit-learn + Streamlit.

🔗 **Live App:** https://aqi-predictor-classifier.streamlit.app

---

## 📌 What it does

Enter pollutant readings for a city and the app will:
- **Predict the numeric AQI value** — using an XGBoost Regressor
- **Classify the air quality category** — using an XGBoost Classifier (Good / Moderate / Poor / Satisfactory / Severe / Very Poor)
- **Show probability scores** for each category
- **Display a health advisory** based on the predicted category

---

## 🖥️ App Preview

| Input Panel | Results Panel |
|---|---|
| City selection + 8 pollutant inputs | AQI value + category + probability bars |

> ⚠️ Best viewed on Chrome, Edge or Firefox

---

## 🧠 Models

| Model | Algorithm | Task |
|---|---|---|
| AQI Predictor | XGBoost Regressor | Predicts numeric AQI value |
| AQI Classifier | XGBoost Classifier | Predicts air quality category |

Both models were tuned using **Optuna** (Bayesian hyperparameter optimization over 100 trials) — significantly better than GridSearch or RandomSearch.

---

## 🛠️ Tech Stack

- **XGBoost** — Regressor + Classifier
- **Scikit-learn** — Pipelines, ColumnTransformer, RobustScaler, OrdinalEncoder, LabelEncoder
- **Optuna** — Hyperparameter tuning
- **Streamlit** — Web app deployment
- **Pandas / NumPy** — Data processing

---

## 📂 Project Structure

```
├── app.py                      # Streamlit web app
├── aqi_predictor.pkl           # Trained XGBoost Regressor pipeline
├── aqi_classifier.pkl          # Trained XGBoost Classifier pipeline + LabelEncoder
├── requirements.txt            # Dependencies    # Optuna tuning script for both models
└── README.md
```

---

## ⚙️ Features

- **Wide layout** — inputs on the left, results on the right in one frame
- **Probability bar chart** — shows confidence for each AQI category
- **Health advisory** — actionable advice based on predicted category
- **Persistent results** — results stay visible when inputs are changed
- **Clean UI** — Streamlit toolbar hidden, dark themed

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/thulasi1212/aqi-app.git
cd aqi-app
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

---

## 📊 Input Features

| Feature | Unit | Description |
|---|---|---|
| PM2.5 | µg/m³ | Fine particulate matter |
| NO | µg/m³ | Nitric oxide |
| NO2 | µg/m³ | Nitrogen dioxide |
| NOx | µg/m³ | Nitrogen oxides |
| CO | mg/m³ | Carbon monoxide |
| SO2 | µg/m³ | Sulphur dioxide |
| O3 | µg/m³ | Ozone |
| Benzene | µg/m³ | Benzene |
| City | — | Indian city name |

---

## 🏷️ AQI Categories

| Category | Health Impact |
|---|---|
| Good | Minimal impact |
| Satisfactory | Minor breathing discomfort to sensitive people |
| Moderate | Breathing discomfort to people with lung/heart disease |
| Poor | Breathing discomfort to most people |
| Very Poor | Respiratory illness on prolonged exposure |
| Severe | Affects healthy people, seriously impacts those with existing diseases |

---

## 📦 Requirements

```
streamlit
scikit-learn
xgboost
pandas
numpy
```

---

## 🙌 Acknowledgements

Dataset sourced from Indian city air quality monitoring data.

---

## 📁 Dataset

The data is stored in the `data/` folder of this repository.

**Source:** Kaggle — [paste your Kaggle dataset link here]

Contains air quality and AQI data across multiple Indian cities with the following pollutants: PM2.5, NO, NO2, NOx, CO, SO2, O3, Benzene and AQI category labels.

---

## 🔗 Links

| | Link |
|---|---|
| 🌐 Live App | https://aqi-predictor-classifier.streamlit.app |
| 📊 Dataset | https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india |
| 👤 GitHub | https://github.com/thulasi1212/aqi-app |

---

*Built as a learning project — feedback and suggestions welcome!*