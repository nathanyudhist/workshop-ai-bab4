# Example 4.21 - SimpleLSTM.py
# A simple LSTM model: Embedding, then LSTM, then one output neuron.

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(None,)),               # sequence of integers with any length
    layers.Embedding(input_dim=100, output_dim=32),
    layers.LSTM(64),
    layers.Dense(1),
])

model.summary()
