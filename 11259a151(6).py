import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Load dataset
data = pd.read_csv("data.csv", header=None)

# Add column names
data.columns = [
    "Student_ID", "Name", "Hours_Studied",
    "Attendance", "Assignments", "Marks", "Result"
]

# Convert columns to numeric values
data["Hours_Studied"] = pd.to_numeric(data["Hours_Studied"])
data["Marks"] = pd.to_numeric(data["Marks"])

# Select features
X = data[["Hours_Studied", "Marks"]]

# Create linkage matrix
linked = linkage(X, method="ward")

# Plot dendrogram
plt.figure(figsize=(8, 5))
dendrogram(linked)
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Students")
plt.ylabel("Euclidean Distance")
plt.show()

# Apply hierarchical clustering
model = AgglomerativeClustering(n_clusters=3)
clusters = model.fit_predict(X)

# Display cluster results
data["Cluster"] = clusters
print("Dataset with Cluster Labels:")
print(data)

# Plot clusters
plt.figure(figsize=(8, 5))
plt.scatter(
    X["Hours_Studied"],
    X["Marks"],
    c=clusters,
    s=100
)
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Hierarchical Clustering")
plt.show()