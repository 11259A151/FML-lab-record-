import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
# Load CSV and assign column names
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

# Remove percentage symbol from Attendance
data["Attendance"] = (
    data["Attendance"]
    .astype(str)
    .str.replace("%", "", regex=False)
)

# Select input features
features = ["Hours_Studied", "Attendance", "Assignments"]
# Convert feature columns to numeric values
for col in features:
    data[col] = pd.to_numeric(data[col], errors="coerce")
# Define input (X) and output (y)
X = data[features]
y = data["Result"].astype(str).str.strip()
# Replace missing values with column means
X = X.fillna(X.mean())
# Remove rows with missing target values
valid_rows = data["Result"].notna()
X = X.loc[valid_rows]
y = y.loc[valid_rows]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("Training Data Size:", len(X_train))
print("Testing Data Size:", len(X_test))
# Create the KNN classifier
model = KNeighborsClassifier(n_neighbors=3)
# Perform 3-fold cross-validation
scores = cross_val_score(
    model,
    X,
    y,
    cv=3)
print("Cross-validation scores:", scores)
print("Mean CV Accuracy:", scores.mean())
# Train the model
model.fit(X_train, y_train)
# Evaluate the model
accuracy = model.score(X_test, y_test)
print("Test Accuracy:", accuracy)
print("Test Accuracy (%):", round(accuracy * 100, 2))
