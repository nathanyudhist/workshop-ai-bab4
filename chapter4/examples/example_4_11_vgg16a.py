# Example 4.11 - VGG16a.py
# Uses VGG16 to classify a single image.
# Usage: python example_4_11_vgg16a.py Elephant.jpg

import sys
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing import image

path = sys.argv[1] if len(sys.argv) > 1 else "Elephant.jpg"

model = VGG16(weights="imagenet")

try:
    img = image.load_img(path, target_size=(224, 224))
except FileNotFoundError:
    raise SystemExit(f"Image file '{path}' not found. Please provide an image first.")

x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
preds = model.predict(x)

print("Top five predictions:")
for _, label, score in decode_predictions(preds, top=5)[0]:
    print(f"   {label:<25} {score * 100:5.1f}%")
