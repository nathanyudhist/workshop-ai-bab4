# Exercise 4.17
# Task: In the feature maps code (Example 4.17 part 6), select the second, third,
# or fourth convolutional layer by changing n = 1 to n = 2 and so on,
# then add a comment.
#
# Usage: python exercise_4_17.py Elephant.jpg
# The program takes the 2nd, 3rd, and 4th convolutional layers (not just the layer index,
# because there are pooling layers in between).

import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

path = sys.argv[1] if len(sys.argv) > 1 else "Elephant.jpg"
try:
    img = image.load_img(path, target_size=(224, 224))
except FileNotFoundError:
    raise SystemExit(f"Image file '{path}' not found. Please provide an image first.")

x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))

vgg = VGG19()
conv_layers = [l for l in vgg.layers if isinstance(l, Conv2D)]


def plot_feature_maps(feature_maps, title, max_maps=64, columns=8):
    count = min(max_maps, feature_maps.shape[3])
    rows = int(np.ceil(count / columns))
    plt.figure(figsize=(16, 2 * rows))
    for i in range(count):
        ax = plt.subplot(rows, columns, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(feature_maps[0, :, :, i], cmap="gray")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


for order in (2, 3, 4):
    layer = conv_layers[order - 1]
    extractor = Model(inputs=vgg.inputs, outputs=layer.output)
    feature_maps = extractor.predict(x, verbose=0)
    print(f"Convolutional layer {order}: {layer.name}, feature maps shape {feature_maps.shape}")
    plot_feature_maps(feature_maps, f"Convolutional layer {order}: {layer.name}")

# RESULT COMMENTS:
# (fill in after running)
# Hint: notice how the image gets smaller and more abstract in deeper layers,
# and which parts of the image light up in each map.
