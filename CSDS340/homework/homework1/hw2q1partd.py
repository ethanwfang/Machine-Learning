import numpy as np

X = np.array([1, 2, 3, 4])
y = np.array([0, 0, 1, 1])

w = 0.1
b = 0.1
eta = 0.1 

def mse(X, y, w, b):
    predictions = w * X + b
    errors = y - predictions
    mse = np.mean(errors**2)
    return mse

initial_loss = mse(X, y, w, b)
for i in range(len(X)):
    y_pred = w * X[i] + b
    
    w = w + eta * (y[i] - y_pred) * X[i]
    b = b + eta * (y[i] - y_pred)

final_loss = mse(X, y, w, b)

print(f"Initial MSE: {initial_loss}")
print(f"Final MSE after one epoch: {final_loss}")
print(f"Updated weight: {w}")
print(f"Updated bias: {b}")
