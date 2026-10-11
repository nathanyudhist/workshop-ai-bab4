# Example 4.12 - MultiClassify.py
# Loads one of several models (VGG16, ResNet50, MobileNet, InceptionV3),
# shows its summary, then classifies an image.
# Change MODEL_NAME to choose the model. Usage: python example_4_12_multiclassify.py Elephant.jpg

import sys
import numpy as np
from tensorflow.keras.applications import vgg16, resnet50, mobilenet, inception_v3
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing import image

MODEL_NAME = "mobilenet"  # choices: vgg16, resnet50, mobilenet, inceptionv3

# name: (model builder, preprocessing module, input size)
MODEL_OPTIONS = {
    "vgg16": (vgg16.VGG16, vgg16, 224),
    "resnet50": (resnet50.ResNet50, resnet50, 224),
    "mobilenet": (mobilenet.MobileNet, mobilenet, 224),
    "inceptionv3": (inception_v3.InceptionV3, inception_v3, 299),
}

builder, module, size = MODEL_OPTIONS[MODEL_NAME]
model = builder(weights="imagenet")
model.summary()

path = sys.argv[1] if len(sys.argv) > 1 else "Elephant.jpg"
try:
    img = image.load_img(path, target_size=(size, size))
except FileNotFoundError:
    raise SystemExit(f"Image file '{path}' not found. Please provide an image first.")

x = module.preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
preds = model.predict(x)

print(f"\nModel: {MODEL_NAME}")
for _, label, score in decode_predictions(preds, top=5)[0]:
    print(f"   {label:<25} {score * 100:5.1f}%")
