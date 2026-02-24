import numpy as np

def sigmoid(x):
    x = np.array(x)   # Convert list to NumPy array
    return 1 / (1 + np.exp(-x))

x = [0, 2, -2]
print(sigmoid(x))