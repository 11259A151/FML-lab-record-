
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv(
    "data.csv",
    header=None,
    names=[
        "Student_ID",
        "Name",
        "Hours_Studied",
        "Attendance",
        "Assignments",
        "Marks",
        "Result"
    ]
)

# Convert Attendance percentage to numeric
data["Attendance"] = (
    data["Attendance"]
    .astype(str)
    .str.replace("%", "", regex=False)
)

# Select input features
X = data[["Hours_Studied", "Attendance", "Assignments"]].apply(
    pd.to_numeric, errors="coerce"
)

# Select target variable
y = data["Result"].astype(str).str.strip()

# Handle missing values
X = X.fillna(X.mean())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create SVM model
model = SVC(kernel="linear")

# Train model
model.fit(X_train, y_train)

# Predict results
y_pred = model.predict(X_test)

# Evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Graph: Actual vs Predicted Results
plt.figure(figsize=(8, 5))

student_index = range(1, len(y_test) + 1)

plt.scatter(
    student_index,
    y_test,
    color="blue",
    marker="o",
    s=100,
    label="Actual Result"
)

plt.scatter(
    student_index,
    y_pred,
    color="red",
    marker="x",
    s=100,
    label="Predicted Result"
)

plt.xlabel("Test Student Index")
plt.ylabel("Result")
plt.title("SVM Classification: Actual vs Predicted")
plt.yticks([0, 1], ["Fail", "Pass"])
plt.ylim(-0.5, 1.5)

# Convert categories into numeric positions for plotting
plt.clf()

plt.figure(figsize=(8, 5))

actual_numeric = [1 if str(v).strip().lower() == "pass" else 0 for v in y_test]
predicted_numeric = [1 if str(v).strip().lower() == "pass" else 0 for v in y_pred]

plt.scatter(
    student_index,
    actual_numeric,
    color="blue",
    marker="o",
    s=100,
    label="Actual Result"
)

plt.scatter(
    student_index,
    predicted_numeric,
    color="red",
    marker="x",
    s=100,
    label="Predicted Result"
)

plt.xlabel("Test Student Index")
plt.ylabel("Result")
plt.yticks([0, 1], ["Fail", "Pass"])
plt.title("SVM: Actual vs Predicted Results")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
