
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load dataset
data = pd.read_csv("data.csv", header=None)

# Add column names
data.columns = [
    "Student_ID",
    "Name",
    "Hours_Studied",
    "Attendance",
    "Assignments",
    "Marks",
    "Result"
]

# Convert Attendance percentage into a number
data["Attendance"] = (
    data["Attendance"]
    .astype(str)
    .str.replace("%", "", regex=False)
)

# Select input features
X = data[
    ["Hours_Studied", "Attendance", "Assignments"]
].copy()

X = X.apply(pd.to_numeric)

# Target variable
y = data["Result"].astype(str).str.strip()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict results
y_pred = model.predict(X_test)

# Display evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

# Graph 1: Actual vs Predicted Results
plt.figure(figsize=(8, 5))

plt.plot(
    range(len(y_test)),
    y_test.to_numpy(),
    marker="o",
    label="Actual Result"
)

plt.plot(
    range(len(y_pred)),
    y_pred,
    marker="x",
    label="Predicted Result"
)

plt.xlabel("Test Student")
plt.ylabel("Result")
plt.title("Random Forest: Actual vs Predicted Results")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Graph 2: Feature Importance Bar Graph
feature_importance = model.feature_importances_

plt.figure(figsize=(8, 5))

plt.bar(
    X.columns,
    feature_importance
)

plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.title("Random Forest - Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
