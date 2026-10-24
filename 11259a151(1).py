import pandas as pd
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("data.csv",
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
print("Original Dataset:")
print(data.head())
print("\nMissing Values:")
print(data.isnull().sum())
data["Attendance"] = (
    data["Attendance"].astype(str).str.replace("%", "", regex=False) )
data["Attendance"] = pd.to_numeric(data["Attendance"], errors="coerce")
features = ["Hours_Studied", "Attendance", "Assignments", "Marks"]
for col in features:
    data[col] = pd.to_numeric(data[col], errors="coerce")
data[features] = data[features].fillna(data[features].mean())
X = data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nScaled Data:")
print(X_scaled)
