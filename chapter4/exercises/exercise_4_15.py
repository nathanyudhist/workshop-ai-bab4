# Exercise 4.15
# Task: In the VGG19 filter visualization code (Example 4.17 part 4), select the second
# convolutional layer by changing n = 1 to n = 2, then add a comment.
#
# In the Keras VGG19 model: layers[0] = input, layers[1] = block1_conv1,
# layers[2] = block1_conv2 (the second convolutional layer).

import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG19

model = VGG19(weights="imagenet")

n = 2
layer = model.layers[n]
print("Selected layer:", layer.name)

filters, biases = layer.get_weights()
s = filters.shape
print("Input channels :", s[2])
print("Filter size    :", s[0], "x", s[1])
print("Filter count   :", s[3])

# Normalize the filter values to 0..1 so they can be displayed
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

n_filters, ix = 4, 1  # show the first 4 filters
plt.figure(figsize=(16, 6))
for i in range(n_filters):
    f = filters[:, :, :, i]
    # This layer has 64 input channels, so only the first 8 channels are shown
    for j in range(8):
        ax = plt.subplot(n_filters, 8, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(f[:, :, j], cmap="gray")
        ix += 1
plt.suptitle(f"Filters in layer {layer.name}")
plt.show()

# RESULT COMMENTS:
# (fill in after running)
# Hint: compare with the first layer (n = 1). In the first layer each filter has
# 3 color channels and often looks like an edge or color detector.
# In the second layer each filter has 64 channels, so the patterns are harder for humans to read.
