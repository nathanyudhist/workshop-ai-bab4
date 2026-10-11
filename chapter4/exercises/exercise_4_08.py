# Exercise 4.8
# Task: Modify Example 4.9 by adding Conv2D, MaxPooling, and Dropout
# before Flatten. Compare the performance.

import time
from tensorflow import keras
from tensorflow.keras import layers

EPOCHS = 5


def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test


def build(extra_block):
    model = keras.Sequential()
    model.add(layers.Input(shape=(28, 28, 1)))
    model.add(layers.Conv2D(32, (5, 5), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    if extra_block:
        model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


x_train, y_train, x_test, y_test = load_mnist()

print("Model                  | Parameters | Time (seconds) | Accuracy | Error")
for name, extra in [("Original", False), ("With the extra block", True)]:
    keras.utils.set_random_seed(42)
    model = build(extra)
    start = time.perf_counter()
    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              epochs=EPOCHS, batch_size=200, verbose=0)
    secs = time.perf_counter() - start
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{name:<22} | {model.count_params():>10,} | {secs:>14.1f} | {acc:.4f}   | {100 - acc * 100:.2f}%")

# RESULT COMMENTS:
# (fill in after running)
