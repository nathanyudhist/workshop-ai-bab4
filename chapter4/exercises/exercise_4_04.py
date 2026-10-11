# Exercise 4.4
# Task: Modify the LeNet program (Example 4.4) with a different number of filters,
# then see the effect on the total number of parameters.

from tensorflow import keras
from tensorflow.keras import layers


def build_lenet(f1, f2):
    return keras.Sequential([
        layers.Input(shape=(32, 32, 1)),
        layers.Conv2D(f1, (5, 5), activation="tanh"),
        layers.AveragePooling2D((2, 2)),
        layers.Conv2D(f2, (5, 5), activation="tanh"),
        layers.AveragePooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(120, activation="tanh"),
        layers.Dense(84, activation="tanh"),
        layers.Dense(10, activation="softmax"),
    ])


print("C1 filters | C2 filters | Total parameters")
for f1, f2 in [(6, 16), (3, 8), (12, 32), (24, 64)]:
    model = build_lenet(f1, f2)
    print(f"{f1:>10} | {f2:>10} | {model.count_params():>16,}")

# RESULT COMMENTS:
# (fill in after running)
# Hint: most parameters are usually in the first Dense layer after Flatten,
# because the Flatten size grows as the number of C2 filters goes up.
