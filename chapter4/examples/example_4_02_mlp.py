# Example 4.2 - MLP.py
# Multi layer perceptron with Scikit-Learn, two inputs and one output (logical AND).

from sklearn.neural_network import MLPClassifier

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 0, 0, 1]

clf = MLPClassifier(
    solver="lbfgs",
    alpha=1e-5,
    hidden_layer_sizes=(5, 2),
    random_state=1,
    max_iter=2000,
)
clf.fit(X, y)

print("Training data predictions:", clf.predict(X))
print("New data predictions     :", clf.predict([[2, 2], [-1, -2]]))
print("Weight matrix shapes     :", [w.shape for w in clf.coefs_])
