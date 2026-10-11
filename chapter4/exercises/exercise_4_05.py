# Exercise 4.5
# Task: Modify the AlexNet program (Example 4.5) with a different number of dense layer units,
# then see the effect on the total number of parameters.

from tensorflow import keras
from tensorflow.keras import layers


def build_alexnet(units):
    return keras.Sequential([
        layers.Input(shape=(227, 227, 3)),
        layers.Conv2D(96, (11, 11), strides=4, activation="relu"),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Conv2D(256, (5, 5), padding="same", activation="relu"),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Conv2D(384, (3, 3), padding="same", activation="relu"),
        layers.Conv2D(384, (3, 3), padding="same", activation="relu"),
        layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Flatten(),
        layers.Dense(units, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(units, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1000, activation="softmax"),
    ])


print("Dense units | Total parameters")
for units in (4096, 2048, 1024, 512):
    keras.backend.clear_session()
    model = build_alexnet(units)
    print(f"{units:>11} | {model.count_params():>16,}")

# RESULT COMMENTS:
# (fill in after running)
# Hint: the Flatten output is 6x6x256 = 9216, so the first dense layer has
# 9216 x units + units parameters. The two large dense layers dominate the total.
