import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.colors import TABLEAU_COLORS

# Settings
os.environ["OMP_NUM_THREADS"] = "1"
plt.style.use('ggplot')

# Output folder next to script
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_images")
os.makedirs(output_dir, exist_ok=True)

# Column keywords to use
TARGET_KEYWORDS = ['notificacoes', 'screen time', 'abertura de apps', 'tempotela', 'minutesawake']

def load_data(filepath):
    """Load and preprocess dataset with specified metric columns"""
    try:
        with open(filepath, encoding='utf-8') as f:
            first_line = f.readline()
            delimiter = "\t" if "\t" in first_line else ","

        df = pd.read_csv(filepath, delimiter=delimiter, encoding='utf-8')

        # Keep only target columns (case insensitive)
        selected_cols = [
            col for col in df.columns
            if any(kw in col.lower() for kw in TARGET_KEYWORDS)
        ]

        if not selected_cols:
            raise ValueError("No matching metric columns found.")

        df = df[selected_cols].select_dtypes(include=[np.number]).fillna(0)
        return df, selected_cols

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None

def find_top_two_k(data, max_k=10):
    """Find the two best k-values using silhouette score"""
    k_scores = {}
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        k_scores[k] = score

    sorted_k = sorted(k_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_k[:2]

def plot_clusters_grid(data, k_range, features):
    """Plot clustering results for k = 2 to 10 with fixed colors and cluster centers as big dots"""
    colors = list(TABLEAU_COLORS.values())
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))

    for idx, k in enumerate(k_range):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        centers = kmeans.cluster_centers_

        ax = axes[idx // 5, idx % 5]
        for cluster_id in range(k):
            cluster_points = data[labels == cluster_id]
            color = colors[cluster_id % len(colors)]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                       color=color, alpha=0.6, s=40)
            ax.scatter(centers[cluster_id, 0], centers[cluster_id, 1],
                       color=color, s=120, alpha=1,
                       edgecolors='black', linewidths=1.5)

        ax.set_title(f'k={k}\nSilhouette={silhouette_score(data, labels):.2f}')
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.grid(True)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "kmeans_notifications_clusters_grid.png")
    plt.savefig(out_path, dpi=150)
    print(f"✅ Saved cluster grid plot: {out_path}")
    plt.close()

def compare_two_best_k(data, k1, k2, features, scores):
    """Compare clustering results for the two best k values with fixed colors and cluster centers as big dots"""
    colors = list(TABLEAU_COLORS.values())
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, k in enumerate([k1, k2]):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        centers = kmeans.cluster_centers_

        ax = axes[idx]
        for cluster_id in range(k):
            cluster_points = data[labels == cluster_id]
            color = colors[cluster_id % len(colors)]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                       color=color, alpha=0.6, s=40)
            ax.scatter(centers[cluster_id, 0], centers[cluster_id, 1],
                       color=color, s=120, alpha=1,
                       edgecolors='black', linewidths=1.5)

        ax.set_title(f'k={k}\nSilhouette={scores[k]:.2f}')
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.grid(True)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"compare_k{k1}_vs_k{k2}.png")
    plt.savefig(out_path, dpi=150)
    print(f"✅ Saved comparison plot of best 2 k: {out_path}")
    plt.close()

def main():
    dataset_dir = r"D:\kmeans\datasets"
    filename = "FinalDataset.csv"
    filepath = os.path.join(dataset_dir, filename)

    if not os.path.exists(filepath):
        print(f"❌ Error: File {filename} not found in {dataset_dir}")
        return

    print(f"📄 Loading dataset from: {filepath}")
    df, features = load_data(filepath)
    if df is None:
        return

    print(f"\n🔎 Found {len(features)} relevant columns:")
    for i, col in enumerate(features, 1):
        print(f"{i}. {col}")

    if df.shape[1] < 2:
        print("❌ Need at least 2 columns for 2D plots.")
        return

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    print("\n📊 Evaluating silhouette scores for k=2 to 10...")
    top_k_pairs = find_top_two_k(scaled_data)
    (k1, s1), (k2, s2) = top_k_pairs

    print("=" * 60)
    print(f"🏆 Top 2 k-values based on Silhouette Score:")
    print(f"1. k={k1}, Score={s1:.3f}")
    print(f"2. k={k2}, Score={s2:.3f}")
    print("=" * 60)

    print("\n📌 Generating grid plot for all k values...")
    plot_clusters_grid(scaled_data[:, :2], range(2, 11), features[:2])

    print(f"\n📌 Comparing top 2 k-values: k={k1} vs k={k2}...")
    scores_dict = {k1: s1, k2: s2}
    compare_two_best_k(scaled_data[:, :2], k1, k2, features[:2], scores_dict)

if __name__ == "__main__":
    main()
