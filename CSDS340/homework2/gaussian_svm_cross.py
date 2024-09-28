# Problem 5
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


X = np.array([[0,1], [-1, 0], [1,0], [0, -1], [0,2], [2,0], [-2, 0], [0, -2]])
y = np.array([1, 1, 1, 1, 0, 0, 0, 0])

clf = SVC(kernel='rbf', gamma='auto', C=1.0)  # Using RBF kernel
clf.fit(X, y)

xx, yy = np.meshgrid(np.linspace(-3, 3, 500), np.linspace(-3, 3, 500))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.75)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
plt.title('Gaussian Kernel SVM Decision Boundary')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()


