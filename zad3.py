# Zadanie 7.1 - Metoda PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

# Ustawienie stylu wykresów
sns.set_theme()
sns.set(font_scale=1.2)

# 1. Wykonaj analizę PCA na własnym zbiorze danych
# Wczytanie przykładowego zbioru danych (wine dataset)
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names

print("Wymiary oryginalnego zbioru danych:", X.shape)
print("Nazwy cech:", feature_names)

# Standaryzacja danych przed PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Wykonanie PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Analiza wyjaśnionej wariancji
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(10, 5))
plt.bar(
    range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.6
)
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, "r-o")
plt.axhline(y=0.8, color="k", linestyle="--")
plt.xlabel("Składowa główna")
plt.ylabel("Współczynnik wyjaśnionej wariancji")
plt.title("Wyjaśniona wariancja przez poszczególne składowe główne")
plt.grid(True)
plt.tight_layout()

# Analiza ładunków PCA (feature loadings)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_matrix = pd.DataFrame(
    loadings,
    columns=[f"PC{i+1}" for i in range(loadings.shape[1])],
    index=feature_names,
)
print("\nMacierz ładunków PCA:")
print(loading_matrix)

# 2. Wykonaj wizualizację skupień dla 2 lub 3 głównych składowych
# Wizualizacja danych w przestrzeni 2D PCA
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", s=70, alpha=0.7)
plt.xlabel(f"PC1 ({explained_variance_ratio[0]:.2%} wariancji)")
plt.ylabel(f"PC2 ({explained_variance_ratio[1]:.2%} wariancji)")
plt.title("Projekcja danych na pierwsze dwie składowe główne")
plt.colorbar(scatter, label="Klasa")
plt.grid(True)
plt.tight_layout()

# Wizualizacja danych w przestrzeni 3D PCA
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(
    X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap="viridis", s=70, alpha=0.7
)
ax.set_xlabel(f"PC1 ({explained_variance_ratio[0]:.2%} wariancji)")
ax.set_ylabel(f"PC2 ({explained_variance_ratio[1]:.2%} wariancji)")
ax.set_zlabel(f"PC3 ({explained_variance_ratio[2]:.2%} wariancji)")
ax.set_title("Projekcja danych na pierwsze trzy składowe główne")
plt.colorbar(scatter, label="Klasa")
plt.tight_layout()

# 3. Porównaj wyniki klasteryzacji przed i po redukcji wymiarowości
# Redukcja wymiarowości do 2 składowych głównych
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# Klasteryzacja na oryginalnych danych
n_clusters = 3  # Znamy prawdziwą liczbę klas
kmeans_original = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_original_labels = kmeans_original.fit_predict(X_scaled)

# Klasteryzacja na zredukowanych danych
kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_pca_labels = kmeans_pca.fit_predict(X_pca_2d)

# Obliczenie współczynników sylwetki
silhouette_original = silhouette_score(X_scaled, kmeans_original_labels)
silhouette_pca = silhouette_score(X_pca_2d, kmeans_pca_labels)

print(
    f"\nWspółczynnik sylwetki dla K-means na oryginalnych danych: {silhouette_original:.4f}"
)
print(f"Współczynnik sylwetki dla K-means na danych po PCA: {silhouette_pca:.4f}")

# Wizualizacja wyników klasteryzacji przed redukcją (ale w przestrzeni PCA dla porównania)
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1], c=kmeans_original_labels, cmap="viridis", s=70, alpha=0.7
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"Klasteryzacja na oryginalnych danych\nSylwetka: {silhouette_original:.4f}")
plt.colorbar(scatter, label="Klaster")
plt.grid(True)

# Wizualizacja wyników klasteryzacji po redukcji
plt.subplot(1, 2, 2)
scatter = plt.scatter(
    X_pca_2d[:, 0], X_pca_2d[:, 1], c=kmeans_pca_labels, cmap="viridis", s=70, alpha=0.7
)
centers = kmeans_pca.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c="red", s=200, alpha=0.9, marker="X")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"Klasteryzacja na danych po PCA\nSylwetka: {silhouette_pca:.4f}")
plt.colorbar(scatter, label="Klaster")
plt.grid(True)
plt.tight_layout()

# Porównanie klastrów z prawdziwymi klasami
comparison_df = pd.DataFrame(
    {
        "Prawdziwa klasa": y,
        "Klaster przed PCA": kmeans_original_labels,
        "Klaster po PCA": kmeans_pca_labels,
    }
)

print("\nMacierz porównania klastrów przed PCA z prawdziwymi klasami:")
print(pd.crosstab(comparison_df["Prawdziwa klasa"], comparison_df["Klaster przed PCA"]))

print("\nMacierz porównania klastrów po PCA z prawdziwymi klasami:")
print(pd.crosstab(comparison_df["Prawdziwa klasa"], comparison_df["Klaster po PCA"]))

plt.show()
