# Zadanie 7.2 - Metody niehierarchiczne (k-means)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

# Metoda "łokcia" do znalezienia optymalnej liczby klastrów
inertia = []
silhouette = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

    # Współczynnik sylwetki (nie obliczamy dla k=1, gdyż wymaga co najmniej 2 klastrów)
    if k > 1:
        labels = kmeans.labels_
        silhouette.append(silhouette_score(X_scaled, labels))

# Wizualizacja metody łokcia
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, "bo-")
plt.xlabel("Liczba klastrów k")
plt.ylabel("Inercja (suma kwadratów odległości)")
plt.title("Metoda łokcia dla określenia optymalnego k")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette, "ro-")
plt.xlabel("Liczba klastrów k")
plt.ylabel("Współczynnik sylwetki")
plt.title("Współczynnik sylwetki dla określenia optymalnego k")
plt.grid(True)
plt.tight_layout()

# Dla zbioru wine, optymalne k to 3 (wiemy, że mamy 3 klasy w danych)
optimal_k = 3
print(f"\nOptymalna liczba klastrów (na podstawie analizy): {optimal_k}")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap="viridis", s=70, alpha=0.7
)
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c="red", s=200, alpha=0.9, marker="X")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"Klastry K-means (k={optimal_k}) w przestrzeni PCA")
plt.colorbar(scatter, label="Klaster")
plt.grid(True)
plt.tight_layout()

# Porównanie klastrów K-means z prawdziwymi klasami
comparison_df = pd.DataFrame({"Prawdziwa klasa": y, "Klaster K-means": kmeans_labels})
confusion_matrix = pd.crosstab(
    comparison_df["Prawdziwa klasa"], comparison_df["Klaster K-means"]
)
print("\nMacierz porównania klastrów K-means z prawdziwymi klasami:")
print(confusion_matrix)

# Wykonanie PCA
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)
explained_variance_ratio = pca_full.explained_variance_ratio_

# Redukcja wymiarowości do 2 składowych
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# K-means na zredukowanych danych
kmeans_pca = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_pca_labels = kmeans_pca.fit_predict(X_pca_2d)

# Obliczenie współczynnika sylwetki dla obu podejść
silhouette_original = silhouette_score(X_scaled, kmeans_labels)
silhouette_pca = silhouette_score(X_pca_2d, kmeans_pca_labels)

print(
    f"\nWspółczynnik sylwetki dla K-means na oryginalnych danych: {silhouette_original:.4f}"
)
print(
    f"Współczynnik sylwetki dla K-means na danych po PCA (2 składowe): {silhouette_pca:.4f}"
)
print(
    f"Wyjaśniona wariancja przez 2 pierwsze składowe PCA: {pca_2d.explained_variance_ratio_.sum():.2%}"
)

# Wizualizacja porównawcza
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap="viridis", s=70, alpha=0.7
)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c="red", s=200, alpha=0.9, marker="X")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"K-means na oryginalnych danych\nSylwetka: {silhouette_original:.4f}")
plt.colorbar(scatter, label="Klaster")
plt.grid(True)

plt.subplot(1, 2, 2)
scatter = plt.scatter(
    X_pca_2d[:, 0], X_pca_2d[:, 1], c=kmeans_pca_labels, cmap="viridis", s=70, alpha=0.7
)
centers = kmeans_pca.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c="red", s=200, alpha=0.9, marker="X")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"K-means na danych po PCA\nSylwetka: {silhouette_pca:.4f}")
plt.colorbar(scatter, label="Klaster")
plt.grid(True)
plt.tight_layout()

# Porównanie zgodności klastrów przed i po PCA
comparison_pca_df = pd.DataFrame(
    {
        "Prawdziwa klasa": y,
        "Klaster na oryginalnych": kmeans_labels,
        "Klaster na PCA": kmeans_pca_labels,
    }
)

print("\nMacierz porównania klastrów przed i po PCA:")
print(
    pd.crosstab(
        comparison_pca_df["Klaster na oryginalnych"],
        comparison_pca_df["Klaster na PCA"],
    )
)

plt.show()
