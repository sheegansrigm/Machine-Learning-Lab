import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import CCA
from sklearn.datasets import load_iris

iris = load_iris()
X1 = iris.data[:, :2]  
X2 = iris.data[:, 2:]  

cca = CCA(n_components=2)
cca.fit(X1, X2)

canonical_coef_X1 = cca.x_weights_
canonical_coef_X2 = cca.y_weights_

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.heatmap(canonical_coef_X1, annot=True, cmap="coolwarm",
            xticklabels=iris.feature_names[:2],
            yticklabels=[f"CCA {i+1}" for i in range(canonical_coef_X1.shape[1])])
plt.title('Canonical Coefficients for X1')

plt.subplot(1, 2, 2)
sns.heatmap(canonical_coef_X2, annot=True, cmap="coolwarm",
            xticklabels=iris.feature_names[2:],
            yticklabels=[f"CCA {i+1}" for i in range(canonical_coef_X2.shape[1])])
plt.title('Canonical Coefficients for X2')

plt.tight_layout()
plt.show()
