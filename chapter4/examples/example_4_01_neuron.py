# Example 4.1 - Neuron.py
# A single neuron with two inputs and one output that learns the logical AND.
# The neuron is trained with simple gradient descent.

import numpy as np

# Training data: AND truth table
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 0, 0, 1], dtype=float)

rng = np.random.default_rng(1)
w = rng.uniform(-1, 1, size=2)
b = 0.0
learning_rate = 0.5


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def neuron(x, w, b):
    return sigmoid(np.dot(x, w) + b)


# Training
for epoch in range(20000):
    out = neuron(X, w, b)
    error = y - out
    gradient = error * out * (1 - out)
    w += learning_rate * X.T @ gradient
    b += learning_rate * gradient.sum()

# Testing
print("Final weights:", w, "bias:", b)
for x in X:
    print(f"Input {x.astype(int)} -> {neuron(x, w, b):.4f}")
