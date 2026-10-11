# Example 4.18 - SimpleRNN.py
# The most basic example of a SimpleRNN layer: feed it random data and look at the output.

import numpy as np
import tensorflow as tf

num_samples = 4   # how many data items at once (batch)
seq_length = 5    # how many time steps
num_features = 1  # features per step

inputs = np.random.random([num_samples, seq_length, num_features]).astype(np.float32)
rnn = tf.keras.layers.SimpleRNN(4)

outputs = rnn(inputs)

print("Input shape :", inputs.shape)
print("Input:\n", inputs)
print("Output shape:", outputs.shape)
print("Output:\n", outputs.numpy())
