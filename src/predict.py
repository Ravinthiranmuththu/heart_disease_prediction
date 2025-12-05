import numpy as np
import pandas as pd
import joblib

# Load saved models and scaler
scaler = joblib.load("scaler.pkl")
svm_model = joblib.load("svm_model.pkl")
rf_model = joblib.load("rf_model.pkl")

# Define the features in the same order as training
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

print("Enter patient data:")

# Collect user input for all features
patient_data = []
for feature in feature_names:
    value = float(input(f"{feature}: "))
    patient_data.append(value)

# Convert to array and scale
patient_array = np.array(patient_data).reshape(1, -1)
patient_scaled = scaler.transform(patient_array)

# Predict using SVM
svm_pred = svm_model.predict(patient_scaled)[0]
svm_prob = svm_model.decision_function(patient_scaled)[0] if hasattr(svm_model, 'decision_function') else None

# Predict using Random Forest
rf_pred = rf_model.predict(patient_scaled)[0]
rf_prob = rf_model.predict_proba(patient_scaled)[0][1]  # probability of disease (class 1)

# Map prediction to human-readable
def interpret(pred):
    return "No Heart Disease" if pred == 0 else "Heart Disease"

print("\n--- Predictions ---")
print(f"SVM Prediction: {interpret(svm_pred)}")
if svm_prob is not None:
    print(f"SVM Confidence Score: {svm_prob:.3f}")
print(f"Random Forest Prediction: {interpret(rf_pred)}")
print(f"Random Forest Probability of Heart Disease: {rf_prob:.3f}")
