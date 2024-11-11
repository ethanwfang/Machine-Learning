import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Problem 5

import pandas as pd

df = pd.read_csv("/Users/efang/Desktop/coding/Intro-to-ML/CSDS340/data/pima-indians-diabetes.csv", header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1]. values

class Perceptron:
    """
        Perception classifier.

        Parameters
        ------------
        eta : float # Learning rate (between 0.0 and 1.0)
        n_iter : int # Passes over the training set
        random_state : int # Random number generator seed for random weight initialization

        Attributes
        -------------
        w_ : 1d-array # Weights after fitting
        b_ : Scalar # Bias unit after fitting
        errors_ : list # Number of misclassifications (updates) in each epoch
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
            Fit training data.

            Parameters 
            -----------
            X : {array-like}, shape = [n_examples, n_features] 
                Training vectors, where n_examples is the number of examples 
                and n_features is the number of features
            y : array-like, shape = [n_examples] 
                Target values

            Returns
            --------
            self : object
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale = 0.01, size = X.shape[1])
        self.b_ = np.float_(0.)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            # Append errors for the epoch after processing all examples
            self.errors_.append(errors)
        
        return self
    
    def net_input(self, X):
        # Calculate net input
        return np.dot(X, self.w_) + self.b_
    
    def predict(self, X):
        # Return class label after unit step
        return np.where(self.net_input(X) >= 0.0, 1, 0)


best_accuracy = 0
best_eta = None
for eta in np.arange(0, 1, 0.001):
    ppn = Perceptron(eta=eta, n_iter=100)

    ppn.fit(X,y)

    y_pred = ppn.predict(X)

    accuracy = accuracy_score(y, y_pred)

    #print(f"Accuracy: {accuracy * 100:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_eta = eta

ppn = Perceptron(eta=0.003, n_iter=100)

ppn.fit(X, y)

y_pred = ppn.predict(X)

accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

plt.plot(range(1, ppn.n_iter + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.title('Perceptron - Errors vs. Epochs')
plt.ylim(bottom = 0)
plt.show()