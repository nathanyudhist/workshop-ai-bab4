# Exercise 4.13
# Task: Modify Example 4.15 (Image-Classifiers library). Enable the different models
# that were commented out, then compare their performance.
#
# Usage: python exercise_4_13.py image.jpg
# Note: the image-classifiers library was made for older Keras versions. If an import
# error appears, use TensorFlow 2.15 or older, or install tf_keras.

import sys
import time
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from classification_models.keras import Classifiers

path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
try:
    img = image.load_img(path, target_size=(224, 224))
except FileNotFoundError:
    raise SystemExit(f"Image file '{path}' not found. Please provide an image first.")
x = np.expand_dims(image.img_to_array(img), axis=0)

# All of these models are enabled (some are commented out in the book)
MODEL_NAMES = [
    "resnet18",
    "resnet34",
    "resnet50",
    "seresnet18",
    "mobilenet",
    "densenet121",
]

print("Model        | Time (ms) | Top prediction")
for name in MODEL_NAMES:
    try:
        Model, preprocess_input = Classifiers.get(name)
        model = Model(input_shape=(224, 224, 3), weights="imagenet", classes=1000)
        batch = preprocess_input(x.copy())
        model.predict(batch, verbose=0)  # warm-up run
        start = time.perf_counter()
        preds = model.predict(batch, verbose=0)
        ms = (time.perf_counter() - start) * 1000
        _, label, score = decode_predictions(preds, top=1)[0][0]
        print(f"{name:<12} | {ms:>9.0f} | {label} ({score * 100:.1f}%)")
    except Exception as e:
        print(f"{name:<12} | failed to load: {e}")

# RESULT COMMENTS:
# (fill in after running)
