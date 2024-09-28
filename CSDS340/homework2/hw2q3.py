# Problem 3

import numpy as np
from sklearn import svm

X = np.array([[1,1], [1,2], [2,1], [0,0], [1,0], [0,1]])
y =np.array([1, 1, 1, 0, 0, 0])

clf = svm.SVC(C=1e8, kernel='linear')
clf.fit(X,y)

w = clf.coef_[0]
b = clf.intercept_[0]

support_vectors = clf.support_vectors_

margin = 2 / np.linalg.norm(2)

print(f"Weight vector (w): {w}")
print(f"Bias (b): {b}")
print(f"Support Vectors: {support_vectors}")
print(f"Margin: {margin:.4f}")