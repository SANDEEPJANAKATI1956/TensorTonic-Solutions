import numpy as np

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    X = np.array(X)
    y = np.array(y)

    N, d = X.shape

    # Initialize parameters
    w = np.zeros(d)
    b = 0.0

    for _ in range(steps):

        # Forward pass
        z = np.dot(X, w) + b
        p = 1 / (1 + np.exp(-z))

        # Gradients
        dw = (1 / N) * np.dot(X.T, (p - y))
        db = (1 / N) * np.sum(p - y)

        # Update
        w -= lr * dw
        b -= lr * db

    return w.tolist(), float(b)