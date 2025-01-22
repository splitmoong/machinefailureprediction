# Machine Failure Prediction

## Project Overview
This project involves predicting machine failures using a dataset of industrial machinery conditions. A Random Forest Classifier is used as the primary model to classify whether a machine will fail.  Techniques like SMOTE were employed to address challenges like class imbalance, EDA for feature selection, and the performance of the model was evaluated.

Dataset: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset


### **Model Comparison**

| Model | Accuracy    | F1 Score    |  Recall |
| :---:   | :---: | :---: |:---: |
| Random Forest   | 0.96 | 0.59 | 0.82 |
| Logistic Regression   |  0.83 | 0.24 | 0.79 |

## Machine Failure Prediction API

This API provides endpoints to:
- Upload a manufacturing dataset.
- Train a machine learning model to predict downtime or defects.
- Make predictions using the trained model.

## Endpoints

### 1. Upload Endpoint
**URL:** `POST /upload`

Upload a CSV file containing manufacturing data.

```bash
curl -X POST -F 'file=@ai4i2020.csv' http://127.0.0.1:5000/upload
```

---

### 2. Train Endpoint
**URL:** `POST /train`

Train the model on the uploaded dataset. Returns performance metrics like accuracy and F1-score.

```bash
curl -X POST http://127.0.0.1:5000/train
```

---

### 3. Predict Endpoint
**URL:** `POST /predict`

Make predictions based on input JSON data. Example input: `{"Temperature": 80, "Run_Time": 120}`.

```bash
curl -X POST -H "Content-Type: application/json" -d '{"Temperature": 80, "Run_Time": 120}' http://127.0.0.1:5000/predict
```

---

## How to Run

1. Install required libraries:
   ```bash
   pip install flask flask-cors pandas scikit-learn
   ```

2. Start the API server:
   ```bash
   python api.py
   ```

3. Use the example `curl` commands above to interact with the API.

---




