
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load CSV and assign column names
data = pd.read_csv(
    "data.csv",
    header=None,
    names=[
        "Student_ID", "Name", "Hours_Studied",
        "Attendance", "Assignments", "Marks", "Result"
    ]
)

# Convert Attendance percentage to a number
data["Attendance"] = data["Attendance"].astype(str).str.replace("%", "", regex=False)

# Select input features
X = data[["Hours_Studied", "Attendance", "Assignments"]].apply(
    pd.to_numeric, errors="coerce"
)

# Select target variable
y = data["Result"].astype(str).str.strip()

# Handle missing numeric values
X = X.fillna(X.mean())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
