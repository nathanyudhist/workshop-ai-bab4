# Example 4.16 - AutoencoderKeras.py
# A simple autoencoder for MNIST images: a 28x28 image is compressed into
# a code of 32 numbers (encoder), then reconstructed again (decoder).

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# Load the data, scale to 0..1
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
print(x_train.shape, x_test.shape)

SHAPE = x_train[0].shape
CODE_SIZE = 32


def build_autoencoder(shape, code_size):
    encoder = keras.Sequential([
        layers.Input(shape=shape),
        layers.Flatten(),
        layers.Dense(code_size),
    ], name="encoder")
    decoder = keras.Sequential([
        layers.Input(shape=(code_size,)),
        layers.Dense(int(np.prod(shape)), activation="sigmoid"),
        layers.Reshape(shape),
    ], name="decoder")
    return encoder, decoder


encoder, decoder = build_autoencoder(SHAPE, CODE_SIZE)

inputs = keras.Input(shape=SHAPE)
autoencoder = keras.Model(inputs, decoder(encoder(inputs)))
autoencoder.compile(optimizer="adamax", loss="mse")
autoencoder.summary()

# Training
history = autoencoder.fit(
    x_train, x_train, epochs=20, batch_size=256,
    validation_data=(x_test, x_test),
)

# Loss plot
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train", "test"], loc="upper right")
plt.show()


def show(img):
    code = encoder.predict(img[None], verbose=0)[0]
    reconstruction = decoder.predict(code[None], verbose=0)[0]

    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(img, cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("Code (32 numbers)")
    plt.imshow(code.reshape(4, 8), cmap="gray")
    plt.subplot(1, 3, 3)
    plt.title("Reconstruction")
    plt.imshow(reconstruction, cmap="gray")
    plt.show()


for i in range(3):
    show(x_test[i])
