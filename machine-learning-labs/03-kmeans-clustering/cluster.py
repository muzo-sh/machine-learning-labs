import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

print("Starting Complete Clustering Practice...\n")
#[part1]

file_names = ['s1.txt', 's2.txt', 's3.txt', 's4.txt']
datasets = []

print("Loading standard files (s1-s4)...")
for file in file_names:
    df = pd.read_csv(file, sep=r'\s+', header=None, names=['X', 'Y'])
    df['X'] = pd.to_numeric(df['X'], errors='coerce')
    df['Y'] = pd.to_numeric(df['Y'], errors='coerce')

    df = df.dropna()

    datasets.append(df)

number_of_clusters = 15

fig1, axs = plt.subplots(2, 2, figsize=(12, 10))
fig1.suptitle("K-Means Clustering on s1 - s4 (15 Clusters)")

for i, (df, ax) in enumerate(zip(datasets, axs.flatten())):
    print(f"Clustering dataset s{i + 1}...")

    #k means model
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=42)
    df['Predicted_Cluster'] = kmeans.fit_predict(df[['X', 'Y']])
    centroids = kmeans.cluster_centers_

    #plot clusters
    ax.scatter(df['X'], df['Y'], c=df['Predicted_Cluster'], cmap='tab20', s=2, alpha=0.5)
    #plot the red
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', s=100, linewidths=3)
    ax.set_title(f"Dataset s{i + 1}")

plt.tight_layout()
print("\n--> Showing s1-s4 graphs. CLOSE THE GRAPH WINDOW to continue to the spiral challenge!")
plt.show()

#part2
print("\nLoading spiral.txt...")
spiral_df = pd.read_csv('spiral.txt', sep=r'\s+', header=None, names=['X', 'Y', 'True_Cluster'])

spiral_df['X'] = pd.to_numeric(spiral_df['X'], errors='coerce')
spiral_df['Y'] = pd.to_numeric(spiral_df['Y'], errors='coerce')
spiral_df['True_Cluster'] = pd.to_numeric(spiral_df['True_Cluster'], errors='coerce')
spiral_df = spiral_df.dropna()

#3clusters
print("Clustering spiral data...")
spiral_kmeans = KMeans(n_clusters=3, random_state=42)
spiral_df['Predicted_Cluster'] = spiral_kmeans.fit_predict(spiral_df[['X', 'Y']])
spiral_centroids = spiral_kmeans.cluster_centers_

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Spiral Dataset: K-Means vs Actual Clusters")

#graph1
ax1.scatter(spiral_df['X'], spiral_df['Y'], c=spiral_df['Predicted_Cluster'], cmap='brg', s=15)
ax1.scatter(spiral_centroids[:, 0], spiral_centroids[:, 1], marker='x', color='black', s=150, linewidths=3)
ax1.set_title("1. What K-Means Guessed")

#graph2
ax2.scatter(spiral_df['X'], spiral_df['Y'], c=spiral_df['True_Cluster'], cmap='brg', s=15)
ax2.set_title("2. The True Answers (From File)")

plt.tight_layout()
print("--> Showing spiral graphs.")
plt.show()

print("\nPractice Complete!")