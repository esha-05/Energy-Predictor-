# Smart Energy Consumption Predictor

An end-to-end Machine Learning system that predicts building energy consumption (heating & cooling load) using multiple regression models and serves predictions via a Flask API.

---

## Project Overview

This project demonstrates the **complete ML lifecycle**:

* Data preprocessing & cleaning
* Feature engineering
* Model training (Linear Regression, Random Forest, XGBoost)
* Model evaluation (MAE, RMSE)
* Anomaly detection
* Visualization dashboard
* Deployment via Flask API

---

## 🎯 Problem Statement

Given building parameters such as:

* Surface area
* Wall area
* Roof area
* Height
* Glazing area

👉 Predict:

* Heating Load (Y1)
* Cooling Load (Y2)

---

## 🧠 What This Project Teaches

* Real-world ML pipeline development
* Feature scaling & preprocessing
* Model comparison and evaluation
* API deployment using Flask
* Visualization using Plotly
* End-to-end system design
---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/energy-predictor.git
cd energy-predictor
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Add dataset

Place your dataset inside:

```
data/ENB2012_data.xlsx
```

---

## ▶️ How to Run the Project

### Step 1: Train the model

```bash
python main.py
```

This will:

* Train multiple models
* Evaluate them
* Save best models in `models/`

---

### Step 2: Start Flask API

```bash
python app.py
```

Server will run on:

```
http://127.0.0.1:5000
```

---

### Step 3: Test API

#### Health Check

```
GET /health
```

Open in browser:

```
http://127.0.0.1:5000/health
```

---

#### Prediction API

```
POST /predict
```

Example request:

```bash
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{
  "compactness": 0.98,
  "surface_area": 514.5,
  "wall_area": 294.0,
  "roof_area": 110.25,
  "height": 7.0,
  "orientation": 2,
  "glazing_area": 0.0,
  "glazing_distribution": 0
}'
```

---

### Example Response

```json
{
  "heating_load": 15.55,
  "cooling_load": 21.33,
  "unit": "kWh/m²"
}
```

---

## 📊 Models Used

| Model             | Description                       |
| ----------------- | --------------------------------- |
| Linear Regression | Baseline model                    |
| Random Forest     | Ensemble model                    |
| XGBoost           | Boosting model (best performance) |

---

## 📈 Evaluation Metrics

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)

---

## 🔍 Anomaly Detection

Implemented using:

* IQR (Interquartile Range)
* Z-score method

Used to detect unusual energy consumption values.

---

## 📊 Visualization

* Actual vs Predicted plots
* Feature importance graphs
* Anomaly detection plots

Built using Plotly for interactive dashboards.

---

## 🌐 API Architecture

```
Client → Flask API → ML Model → Prediction → JSON Response
```
## 🧪 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Plotly
* Flask

---

## 🚀 Future Improvements

* Add React dashboard
* Deploy on cloud (Render / AWS)
* Add real-time prediction
* Improve model accuracy with tuning

