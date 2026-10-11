# Example 4.5 - AlexNet.py
# AlexNet style model with Keras, for 224x224 images and 17 classes.
# The model is only built and summarized, not trained.

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(224, 224, 3)),
    # Convolutional blocks
    layers.Conv2D(96, (11, 11), strides=4, activation="relu"),
    layers.MaxPooling2D((3, 3), strides=2),
    layers.Conv2D(256, (5, 5), padding="same", activation="relu"),
    layers.MaxPooling2D((3, 3), strides=2),
    layers.Conv2D(384, (3, 3), padding="same", activation="relu"),
    layers.Conv2D(384, (3, 3), padding="same", activation="relu"),
    layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D((3, 3), strides=2),
    # Fully connected blocks
    layers.Flatten(),
    layers.Dense(4096, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(4096, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(17, activation="softmax"),
])

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# To train: model.fit(...), to test: model.evaluate(...)
