
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset and assign column names
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

# Select features for clustering
X = data[["Hours_Studied", "Marks"]].apply(
    pd.to_numeric, errors="coerce"
)

# Handle missing values
X = X.fillna(X.mean())

print("Original Dataset:")
print(data.head())

# Create K-Means model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# Fit model and predict clusters
clusters = kmeans.fit_predict(X)

# Add cluster labels to dataset
data["Cluster"] = clusters

print("\nDataset with Clusters:")
print(data[["Student_ID", "Hours_Studied", "Marks", "Cluster"]])

# Plot clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    X["Hours_Studied"],
    X["Marks"],
    c=clusters,
    cmap="viridis",
    s=100,
    label="Students"
)

# Plot cluster centers
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    s=250,
    color="red",
    label="Centroids"
)

plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("K-Means Clustering of Students")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
