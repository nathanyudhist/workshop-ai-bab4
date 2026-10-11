# Exercise 4.3
# Task: Modify Example 4.3 (Keras MLP, breast cancer) so that the dense layer
# has 5 neurons instead of 10. Compare training time and accuracy.

import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def run(units, epochs=50):
    keras.utils.set_random_seed(42)
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(units, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    start = time.perf_counter()
    model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=0)
    elapsed = time.perf_counter() - start
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    return model.count_params(), elapsed, acc


print("Neurons | Parameters | Training time (seconds) | Test accuracy")
for units in (10, 5):
    params, secs, acc = run(units)
    print(f"{units:>7} | {params:>10} | {secs:>23.2f} | {acc:.4f}")

# RESULT COMMENTS:
# (fill in after running)
# Common expectation: parameters drop by almost half, training time is about the same,
# accuracy is slightly different or the same because the data is simple. Check with your results.
