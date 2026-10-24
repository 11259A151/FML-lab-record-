import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
# Load CSV and assign column names
data = pd.read_csv(
    "data.csv",
    header=None,
    names=["Student_ID", "Name", "Hours_Studied","Attendance", "Assignments", "Marks", "Result" ]
data["Attendance"] = data ["Attendance"].astype(str).str.replace("%", "", regex=False)
features = ["Hours_Studied", "Attendance", "Assignments"]
for col in features:
    data[col] = pd.to_numeric(data[col], errors="coerce")
X = data[features]
y = data["Result"].astype(str).str.strip()
X = X.fillna(X.mean())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("Training Data Size:", len(X_train))
print("Testing Data Size:", len(X_test))
# KNN model
model = KNeighborsClassifier(n_neighbors=3)
# Cross-validation (use fewer folds for a small dataset)
scores = cross_val_score(model, X, y, cv=3)
print("Cross-validation scores:", scores)
print("Mean CV Accuracy:", scores.mean())
# Train and evaluate
model.fit(X_train, y_train)
print("Test Accuracy:", model.score(X_test, y_test))
