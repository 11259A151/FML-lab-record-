
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset (CSV has no header row)
data = pd.read_csv("data.csv", header=None)

# Assign column names
data.columns = [
    "Student_ID",
    "Name",
    "Hours_Studied",
    "Attendance",
    "Assignments",
    "Marks",
    "Result"
]

# Convert attendance percentage to numeric
data["Attendance"] = (
    data["Attendance"].astype(str).str.replace("%", "", regex=False)
)

# Input features
X = data[["Hours_Studied", "Attendance", "Assignments"]].copy()
X = X.apply(pd.to_numeric)

# Target variable
y = data["Result"].astype(str).str.strip()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Create AdaBoost model
model = AdaBoostClassifier(n_estimators=50, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Feature Importance Bar Graph
importance = model.feature_importances_

plt.figure(figsize=(8, 5))
plt.bar(X.columns, importance)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("AdaBoost Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
