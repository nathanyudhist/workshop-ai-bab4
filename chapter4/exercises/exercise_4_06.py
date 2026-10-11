# Exercise 4.6
# Task: Modify Example 4.7 by adding one Conv2D, MaxPooling, and Dropout block
# before Flatten. Compare the number of parameters.

from tensorflow import keras
from tensorflow.keras import layers


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
    model.add(layers.Dense(2, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


original = build(False)
modified = build(True)

print("=== Original model ===")
original.summary()
print("\n=== Model with the extra block ===")
modified.summary()

print(f"\nOriginal model parameters : {original.count_params():,}")
print(f"Modified model parameters : {modified.count_params():,}")
print(f"Difference                : {modified.count_params() - original.count_params():,}")

# RESULT COMMENTS:
# (fill in after running)
# Hint: the extra block adds Conv2D parameters, but the extra pooling makes the
# Flatten output smaller, so Dense(128) actually has fewer parameters.
