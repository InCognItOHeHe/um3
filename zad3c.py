# Zadanie 7.3 - Metody hierarchiczne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Ustawienie stylu wykresów
sns.set_theme()
sns.set(font_scale=1.2)

# Wczytanie przykładowego zbioru danych (wine dataset)
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names

print("Wymiary zbioru danych:", X.shape)
print("Liczba klas w zbiorze danych:", len(np.unique(y)))

# Standaryzacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Redukcja wymiarowości dla wizualizacji
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

methods = ["ward", "complete", "single", "average"]

plt.figure(figsize=(20, 15))
for i, method in enumerate(methods):
    # Obliczenie macierzy połączeń
    Z = linkage(X_scaled, method=method)

    # Wygenerowanie dendrogramu
    plt.subplot(2, 2, i + 1)
    dendrogram(
        Z, orientation="top", leaf_rotation=90, color_threshold=0.7 * max(Z[:, 2])
    )
    plt.title(f"Dendrogram dla metody {method}")
    plt.xlabel("Indeks próbki")
    plt.ylabel("Odległość")

plt.tight_layout()

n_clusters_list = [2, 3, 4, 5]
best_method = "ward"  # wybór na podstawie poprzedniej analizy

plt.figure(figsize=(15, 10))
for i, n_clusters in enumerate(n_clusters_list):
    # Hierarchiczna klasteryzacja
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=best_method)
    hierarchical_labels = hierarchical.fit_predict(X_scaled)

    # Wizualizacja klastrów
    plt.subplot(2, 2, i + 1)
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, cmap="viridis", s=70, alpha=0.7
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"Hierarchiczna klasteryzacja: {n_clusters} klastrów")
    plt.colorbar(scatter, label="Klaster")
    plt.grid(True)

plt.tight_layout()

# Porównanie klasteryzacji hierarchicznej z K-means
optimal_n_clusters = 3  # Dostosuj na podstawie analizy dendrogramu
hierarchical = AgglomerativeClustering(
    n_clusters=optimal_n_clusters, linkage=best_method
)
hierarchical_labels = hierarchical.fit_predict(X_scaled)

kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Obliczenie współczynnika sylwetki dla obu metod
silhouette_hierarchical = silhouette_score(X_scaled, hierarchical_labels)
silhouette_kmeans = silhouette_score(X_scaled, kmeans_labels)

print(
    f"\nWspółczynnik sylwetki dla klasteryzacji hierarchicznej: {silhouette_hierarchical:.4f}"
)
print(f"Współczynnik sylwetki dla K-means: {silhouette_kmeans:.4f}")

# Porównanie macierzy klastrów
comparison_df = pd.DataFrame(
    {
        "Prawdziwa klasa": y,
        "Klaster hierarchiczny": hierarchical_labels,
        "Klaster K-means": kmeans_labels,
    }
)

print("\nMacierz porównania klastrów hierarchicznych z prawdziwymi klasami:")
print(
    pd.crosstab(
        comparison_df["Prawdziwa klasa"], comparison_df["Klaster hierarchiczny"]
    )
)

print("\nMacierz porównania klastrów K-means z prawdziwymi klasami:")
print(pd.crosstab(comparison_df["Prawdziwa klasa"], comparison_df["Klaster K-means"]))

print("\nMacierz porównania klastrów hierarchicznych z K-means:")
print(
    pd.crosstab(
        comparison_df["Klaster hierarchiczny"], comparison_df["Klaster K-means"]
    )
)

# Wizualizacja porównawcza klasteryzacji hierarchicznej i K-means
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, cmap="viridis", s=70, alpha=0.7
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"Klasteryzacja hierarchiczna\nSylwetka: {silhouette_hierarchical:.4f}")
plt.colorbar(scatter, label="Klaster")
plt.grid(True)

plt.subplot(1, 2, 2)
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap="viridis", s=70, alpha=0.7
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"Klasteryzacja K-means\nSylwetka: {silhouette_kmeans:.4f}")
plt.colorbar(scatter, label="Klaster")
plt.grid(True)
plt.tight_layout()

pca_for_hierarchical = PCA(n_components=5)  # Zachowujemy 5 pierwszych składowych
X_pca_for_hierarchical = pca_for_hierarchical.fit_transform(X_scaled)

# Hierarchiczna klasteryzacja na danych po PCA
hierarchical_pca = AgglomerativeClustering(
    n_clusters=optimal_n_clusters, linkage=best_method
)
hierarchical_pca_labels = hierarchical_pca.fit_predict(X_pca_for_hierarchical)

# Obliczenie współczynnika sylwetki po PCA
silhouette_hierarchical_pca = silhouette_score(
    X_pca_for_hierarchical, hierarchical_pca_labels
)

print(
    f"\nWspółczynnik sylwetki dla klasteryzacji hierarchicznej po PCA: {silhouette_hierarchical_pca:.4f}"
)
print(
    f"Procent wyjaśnionej wariancji przez 5 pierwszych składowych PCA: {sum(pca_for_hierarchical.explained_variance_ratio_):.2%}"
)

# Porównanie wyników klasteryzacji hierarchicznej przed i po PCA
comparison_pca_df = pd.DataFrame(
    {
        "Prawdziwa klasa": y,
        "Klaster hierarchiczny - oryginalne": hierarchical_labels,
        "Klaster hierarchiczny - PCA": hierarchical_pca_labels,
    }
)

print("\nMacierz porównania klastrów hierarchicznych przed i po PCA:")
print(
    pd.crosstab(
        comparison_pca_df["Klaster hierarchiczny - oryginalne"],
        comparison_pca_df["Klaster hierarchiczny - PCA"],
    )
)

# Wizualizacja końcowa - porównanie wszystkich metod
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", s=70, alpha=0.7)
plt.title("Oryginalne klasy")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(scatter, label="Klasa")
plt.grid(True)

plt.subplot(2, 2, 2)
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap="viridis", s=70, alpha=0.7
)
plt.title("K-means")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(scatter, label="Klaster")
plt.grid(True)

plt.subplot(2, 2, 3)
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, cmap="viridis", s=70, alpha=0.7
)
plt.title("Klasteryzacja hierarchiczna")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(scatter, label="Klaster")
plt.grid(True)

plt.subplot(2, 2, 4)
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1], c=hierarchical_pca_labels, cmap="viridis", s=70, alpha=0.7
)
plt.title("Klasteryzacja hierarchiczna po PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(scatter, label="Klaster")
plt.grid(True)

plt.tight_layout()
plt.show()
