# Example 4.19 - SimpleRNN2.py
# RNN model with one SimpleRNN layer, one Dense layer, and softmax.
# It only builds the model and shows its summary.

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(49, 1)),
    layers.SimpleRNN(50, return_sequences=False),
    layers.Dense(46),
    layers.Activation("softmax"),
])

model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)
model.summary()
