# Problem 2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd

data = {
    'Input1': [0, 0, 0, 0, 1, 1, 1, 1],
    'Input2': [0, 0, 1, 1, 0, 0, 1, 1],
    'Input3': [0, 1, 0, 1, 0, 1, 0, 1],
    'Output': [1, 0, 0, 1, 0, 1, 1, 0]  
}

df = pd.DataFrame(data)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

class Perceptron:
    def __init__(self, eta=0.1, n_iter=10, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        self.w_ = np.full(X.shape[1], 0.1)
        self.b_ = np.float_(0.1)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

y_pred = ppn.predict(X)
accuracy = accuracy_score(y, y_pred)

print(f"Accuracy after 10 epochs: {accuracy * 100:.2f}%")
print(f"Weights after 10 epochs: {ppn.w_}")
print(f"Bias after 10 epochs: {ppn.b_}")

plt.plot(range(1, ppn.n_iter + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.title('Perceptron - Errors vs. Epochs (Parity Problem)')
plt.ylim(bottom=0)
plt.show()

# TODO: SUBMIT THE DIAGRAM
