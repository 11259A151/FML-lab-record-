
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset without treating the first row as headers
data = pd.read_csv("data.csv", header=None)

# Assign the correct column names
data.columns = [
    'Student_ID', 'Name', 'Hours_Studied',
    'Attendance', 'Assignments', 'Marks', 'Result'
]

# Convert numerical columns to numeric values
data['Hours_Studied'] = pd.to_numeric(data['Hours_Studied'])
data['Attendance'] = data['Attendance'].astype(str).str.replace('%', '', regex=False)
data['Attendance'] = pd.to_numeric(data['Attendance'])
data['Assignments'] = pd.to_numeric(data['Assignments'])

# Select input features and target
X = data[['Hours_Studied', 'Attendance', 'Assignments']]
y = data['Result']

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Display results
print("Training Data:")
print(X_train)

print("\nTesting Data:")
print(X_test)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))
