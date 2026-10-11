# Example 4.4 - LeNet.py
# LeNet-5 style architecture with Keras. It only builds the model and shows its summary.
# Expected total parameters: 81,194

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(32, 32, 1)),
    layers.Conv2D(6, (3, 3), activation="relu"),
    layers.AveragePooling2D((2, 2)),
    layers.Conv2D(16, (3, 3), activation="relu"),
    layers.AveragePooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(120, activation="relu"),
    layers.Dense(84, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

model.summary()
print("Total parameters:", f"{model.count_params():,}")
