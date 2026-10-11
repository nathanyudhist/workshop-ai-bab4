# Exercise 4.10
# Task: Modify Example 4.11 so that it uses VGG19 to classify an image,
# then compare its performance with VGG16.
#
# Usage: python exercise_4_10.py image.jpg
# Without an argument, the program looks for test.jpg in the same folder.

import sys
import time
import numpy as np
from tensorflow.keras.applications import vgg16, vgg19
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing import image

path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
try:
    img = image.load_img(path, target_size=(224, 224))
except FileNotFoundError:
    raise SystemExit(f"Image file '{path}' not found. Please provide an image first.")

x = np.expand_dims(image.img_to_array(img), axis=0)


def evaluate(name, builder, module):
    model = builder(weights="imagenet")
    batch = module.preprocess_input(x.copy())
    model.predict(batch, verbose=0)  # warm-up run, not counted
    times = []
    for _ in range(5):
        start = time.perf_counter()
        preds = model.predict(batch, verbose=0)
        times.append(time.perf_counter() - start)
    top3 = decode_predictions(preds, top=3)[0]
    return name, model.count_params(), np.mean(times) * 1000, top3


results = [
    evaluate("VGG16", vgg16.VGG16, vgg16),
    evaluate("VGG19", vgg19.VGG19, vgg19),
]

for name, params, ms, top3 in results:
    print(f"\n{name}: {params:,} parameters, average {ms:.0f} ms per image")
    for _, label, score in top3:
        print(f"   {label:<25} {score * 100:5.1f}%")

# RESULT COMMENTS:
# (fill in after running)
