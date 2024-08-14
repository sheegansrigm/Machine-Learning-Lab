from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target

selector = SelectKBest(chi2, k=3).fit(X, y)
selected_features_indices = selector.get_support(indices=True)
selected_features = [iris.feature_names[i] for i in selected_features_indices]
X_new = selector.transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Original shape:", X.shape)
print("Shape after feature selection:", X_new.shape)
print("Shape after feature transformation:", X_scaled.shape)
print("Shape after feature extraction:", X_pca.shape)
print("Selected features:", selected_features)

plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_new[:, 0], X_new[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.title('Feature Selection')

plt.subplot(1, 3, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.title('Feature Transformation')

plt.subplot(1, 3, 3)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.title('Feature Extraction (PCA)')

plt.tight_layout()
plt.show()
