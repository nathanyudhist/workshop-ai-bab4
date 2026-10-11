# Example 4.17 - CNN Visualize Filters
# In the book this is a notebook with 6 parts, here combined into one script.
# Contents: VGG19 summary, layer list, filter shapes, filter visualization, and feature maps.
# Usage: python example_4_17_cnn_visualize_filters.py Elephant.jpg

import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# ---- Part 1: load the model and see its summary
model = VGG19()
model.summary()

# ---- Part 2: list all layers with their index
for idx, layer in enumerate(model.layers):
    print(idx, layer.name)

# ---- Part 3: filter and bias shapes of each convolutional layer
for idx, layer in enumerate(model.layers):
    if "conv" in layer.name:
        filters, biases = layer.get_weights()
        print(idx, layer.name, filters.shape, biases.shape)

# ---- Part 4: show some filters from one layer
n = 1  # layer index (1 = block1_conv1)
filters, biases = model.layers[n].get_weights()
height, width, channels, count = filters.shape  # Keras filter shape: (height, width, channels, count)
print("Layer          :", model.layers[n].name)
print("Input channels :", channels)
print("Filter size    :", height, "x", width)
print("Filter count   :", count)

# Normalize the values to 0..1 so they can be displayed
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

n_filters = 4  # to show all of them, replace with: count
plt.figure(figsize=(8, 8))
ix = 1
for i in range(n_filters):
    f = filters[:, :, :, i]
    for j in range(channels):
        ax = plt.subplot(n_filters, channels, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(f[:, :, j], cmap="gray")
        ix += 1
plt.show()


# ---- Part 5: function to show feature maps
def plot_feature_maps(feature_maps, columns=8):
    num_maps = feature_maps.shape[3]
    rows = int(np.ceil(num_maps / columns))
    plt.figure(figsize=(16, 2 * rows))
    for i in range(num_maps):
        ax = plt.subplot(rows, columns, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(feature_maps[0, :, :, i], cmap="gray")
    plt.tight_layout()
    plt.show()


# ---- Part 6: feature maps of one layer for one image
path = sys.argv[1] if len(sys.argv) > 1 else "Elephant.jpg"
try:
    img = image.load_img(path, target_size=(224, 224))
except FileNotFoundError:
    raise SystemExit(f"Image file '{path}' not found. Please provide an image first.")

n = 1  # layer to visualize
extractor = Model(inputs=model.inputs, outputs=model.layers[n].output)
extractor.summary()

x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
feature_maps = extractor.predict(x)
print("Feature maps shape:", feature_maps.shape)
plot_feature_maps(feature_maps)
