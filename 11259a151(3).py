
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset without headers
data = pd.read_csv(
    "data.csv",
    header=None,
    names=[
        "Student_ID", "Name", "Hours_Studied",
        "Attendance", "Assignments", "Marks", "Result"
    ]
)

# Input feature
X = data[["Hours_Studied"]]

# Target variable
y = data["Marks"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict marks
y_pred = model.predict(X_test)

print("Actual Marks:", y_test.tolist())
print("Predicted Marks:", y_pred.round(2).tolist())

# Evaluate the model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Plot the regression line
plt.scatter(X, y, color="blue", label="Actual Marks")

plt.plot(
    X,
    model.predict(X),
    color="red",
    label="Regression Line"
)

plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Linear Regression: Hours Studied vs Marks")
plt.legend()
plt.show()
