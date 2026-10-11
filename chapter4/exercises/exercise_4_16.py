# Exercise 4.16
# Task: In the filter visualization code, show all the filters by changing
# n_filters, ix = 4, 1 to n_filters, ix = s[3], 1, then add a comment.

import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG19

model = VGG19(weights="imagenet")

n = 1
layer = model.layers[n]
print("Selected layer:", layer.name)

filters, biases = layer.get_weights()
s = filters.shape
print("Color channels :", s[2])
print("Filter size    :", s[0], "x", s[1])
print("Filter count   :", s[3])

f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

n_filters, ix = s[3], 1  # all filters (64)
channels = s[2]          # 3 color channels
plt.figure(figsize=(6, n_filters * 0.8))
for i in range(n_filters):
    f = filters[:, :, :, i]
    for j in range(channels):
        ax = plt.subplot(n_filters, channels, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(f[:, :, j], cmap="gray")
        ix += 1

plt.tight_layout()
plt.savefig("exercise_4_16_all_filters.png", dpi=100)
print("Image saved to exercise_4_16_all_filters.png")
plt.show()

# RESULT COMMENTS:
# (fill in after running)
