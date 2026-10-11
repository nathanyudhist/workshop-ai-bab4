# Exercise 4.7
# Task: Modify Example 4.9 (CNN for MNIST).
# a) Change 32 to 64 in Conv2D (meaning the number of filters) and compare performance.
# b) Change the filter size (5,5) to (3,3) and compare performance.

import time
from tensorflow import keras
from tensorflow.keras import layers

EPOCHS = 5  # raise to 10 to match the book


def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test


def build(filters, kernel):
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(filters, kernel, activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


x_train, y_train, x_test, y_test = load_mnist()

configs = [
    ("Original: 32 filters, 5x5 kernel", 32, (5, 5)),
    ("64 filters, 5x5 kernel", 64, (5, 5)),
    ("32 filters, 3x3 kernel", 32, (3, 3)),
]

results = []
for name, filters, kernel in configs:
    keras.utils.set_random_seed(42)
    model = build(filters, kernel)
    start = time.perf_counter()
    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              epochs=EPOCHS, batch_size=200, verbose=0)
    secs = time.perf_counter() - start
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    results.append((name, model.count_params(), secs, acc))

print("Configuration                    | Parameters | Time (seconds) | Accuracy | Error")
for name, params, secs, acc in results:
    print(f"{name:<32} | {params:>10,} | {secs:>14.1f} | {acc:.4f}   | {100 - acc * 100:.2f}%")

# RESULT COMMENTS:
# (fill in after running)
