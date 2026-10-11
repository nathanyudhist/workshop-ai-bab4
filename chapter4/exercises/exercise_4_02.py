# Exercise 4.2
# Task: Modify Example 4.2 (Scikit-Learn MLP, logical AND) so that it has three inputs.

from sklearn.neural_network import MLPClassifier

# All combinations of 3 inputs (8 rows)
X = [[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)]
# Target: AND of the three inputs (1 only when all are 1)
y = [int(a and b and c) for a, b, c in X]

clf = MLPClassifier(
    hidden_layer_sizes=(5,),
    activation="logistic",
    solver="lbfgs",
    max_iter=2000,
    random_state=1,
)
clf.fit(X, y)

pred = clf.predict(X)
print("Input      | Target | Prediction")
for xi, yi, pi in zip(X, y, pred):
    print(f"{xi} |   {yi}    |   {pi}")
print("Accuracy:", clf.score(X, y))

# RESULT COMMENTS:
# (fill in after running)
