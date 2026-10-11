# Example 4.9 - MNIST2.py
# A simple CNN to classify MNIST handwritten digits.

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# Load the data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Show four sample images
plt.figure(figsize=(8, 2.5))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.axis("off")
plt.show()

# Reshape the data to [samples][height][width][channels] and scale to 0..1
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255

# One hot encoding of the labels
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Model
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (5, 5), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
])
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# Training and evaluation
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200)
scores = model.evaluate(x_test, y_test, verbose=0)
print(f"CNN error: {100 - scores[1] * 100:.2f}%")
