import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
y = 2 * y - 1
svm = SVC(kernel='linear')
svm.fit(X, y)
w = svm.coef_[0]
b = svm.intercept_[0]
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')
x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_vals = -(w[0] / w[1]) * x_vals - b / w[1]
plt.plot(x_vals, y_vals, 'k-', label='Decision Boundary')
y_vals_margin_1 = y_vals + 1 / w[1]
y_vals_margin_2 = y_vals - 1 / w[1]
plt.plot(x_vals, y_vals_margin_1, 'k--', label='Margin')
plt.plot(x_vals, y_vals_margin_2, 'k--')
plt.legend()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary and Support Vectors')
plt.show()