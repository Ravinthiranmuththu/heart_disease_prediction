import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # <-- for saving models

# 1. Load dataset
df = pd.read_csv("data/heart.csv")  # update with your CSV file path

# 2. Handle invalid entries and convert numeric columns
numeric_cols = ['age','trestbps','chol','thalach','oldpeak','ca','thal']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# 3. Fill missing numeric values with mean
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# 4. Fill missing categorical values with mode
cat_cols = ['sex','cp','fbs','restecg','exang','slope']
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 5. Binary target
df['target_binary'] = df['target'].apply(lambda x: 0 if x == 0 else 1)

# 6. Features & target
X = df.drop(columns=['target', 'target_binary'])
y = df['target_binary']

# 7. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved as 'scaler.pkl'")

# 9. Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("Class distribution after SMOTE:", np.bincount(y_train_res))

# ----------------- SVM -----------------
svm_param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1, 0.001],
    'kernel': ['rbf', 'linear']
}
svm_grid = GridSearchCV(SVC(class_weight='balanced'), svm_param_grid, cv=5)
svm_grid.fit(X_train_res, y_train_res)
y_pred_svm = svm_grid.predict(X_test)
svm_acc = accuracy_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm, average='weighted')
print("\nSVM Best Parameters:", svm_grid.best_params_)
print("SVM Accuracy:", svm_acc)
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues')
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix_svm.png")
plt.close()
print("SVM confusion matrix saved as 'confusion_matrix_svm.png'")

# Save SVM model
joblib.dump(svm_grid.best_estimator_, "svm_model.pkl")
print("SVM model saved as 'svm_model.pkl'")

# ----------------- Random Forest -----------------
rf_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 3, 5, 7],
    'min_samples_split': [2, 5, 10]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5)
rf_grid.fit(X_train_res, y_train_res)
y_pred_rf = rf_grid.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf, average='weighted')
print("\nRandom Forest Best Parameters:", rf_grid.best_params_)
print("Random Forest Accuracy:", rf_acc)
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix_rf.png")
plt.close()
print("Random Forest confusion matrix saved as 'confusion_matrix_rf.png'")

# Save Random Forest model
joblib.dump(rf_grid.best_estimator_, "rf_model.pkl")
print("Random Forest model saved as 'rf_model.pkl'")

# ----------------- Compare Models -----------------
print("\nModel Comparison:")
if rf_acc > svm_acc and rf_f1 > svm_f1:
    print("Random Forest performs better than SVM based on accuracy and weighted F1-score.")
elif svm_acc > rf_acc and svm_f1 > rf_f1:
    print("SVM performs better than Random Forest based on accuracy and weighted F1-score.")
else:
    print("The models perform comparably. Check accuracy and F1-score to choose.")
print(f"SVM - Accuracy: {svm_acc:.4f}, Weighted F1: {svm_f1:.4f}")
print(f"Random Forest - Accuracy: {rf_acc:.4f}, Weighted F1: {rf_f1:.4f}")
