# Exercise 4.14
# Task: Modify Example 4.15 so that it uses if-else to select the classification model.
#
# Usage: python exercise_4_14.py image.jpg resnet50
# Model choices: resnet18, resnet34, resnet50, mobilenet, densenet121

import sys
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from classification_models.keras import Classifiers

path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
choice = sys.argv[2] if len(sys.argv) > 2 else "resnet18"

# Model selection with if-elif-else
if choice == "resnet18":
    Model, preprocess_input = Classifiers.get("resnet18")
elif choice == "resnet34":
    Model, preprocess_input = Classifiers.get("resnet34")
elif choice == "resnet50":
    Model, preprocess_input = Classifiers.get("resnet50")
elif choice == "mobilenet":
    Model, preprocess_input = Classifiers.get("mobilenet")
elif choice == "densenet121":
    Model, preprocess_input = Classifiers.get("densenet121")
else:
    raise SystemExit(
        f"Unknown model '{choice}'. "
        "Choose: resnet18, resnet34, resnet50, mobilenet, densenet121"
    )

try:
    img = image.load_img(path, target_size=(224, 224))
except FileNotFoundError:
    raise SystemExit(f"Image file '{path}' not found. Please provide an image first.")

x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
model = Model(input_shape=(224, 224, 3), weights="imagenet", classes=1000)

preds = model.predict(x, verbose=0)
print(f"Model: {choice}")
for _, label, score in decode_predictions(preds, top=3)[0]:
    print(f"   {label:<25} {score * 100:5.1f}%")

# RESULT COMMENTS:
# (fill in after running)
