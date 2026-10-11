# Exercise 4.12
# Task: Modify the multi-model webcam program (Example 4.14) and add DenseNet.
# Compare the performance.
#
# Controls: press N to switch to the next model, ESC to quit.

import time
import cv2
import numpy as np
from tensorflow.keras.applications import (
    vgg16, resnet50, mobilenet, inception_v3, densenet,
)
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# name: (model builder, preprocessing module, input size)
MODEL_LIST = {
    "VGG16": (vgg16.VGG16, vgg16, 224),
    "ResNet50": (resnet50.ResNet50, resnet50, 224),
    "MobileNet": (mobilenet.MobileNet, mobilenet, 224),
    "InceptionV3": (inception_v3.InceptionV3, inception_v3, 299),
    "DenseNet121": (densenet.DenseNet121, densenet, 224),  # added for this exercise
}

print("Loading models, please wait...")
loaded_models = {}
for name, (builder, module, size) in MODEL_LIST.items():
    loaded_models[name] = (builder(weights="imagenet"), module, size)
model_names = list(loaded_models.keys())
active = 0

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise SystemExit("Cannot open the webcam.")

while True:
    ok, frame = camera.read()
    if not ok:
        break

    name = model_names[active]
    model, module, size = loaded_models[name]

    rgb = cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2RGB)
    batch = module.preprocess_input(np.expand_dims(rgb.astype("float32"), axis=0))

    start = time.perf_counter()
    preds = model.predict(batch, verbose=0)
    ms = (time.perf_counter() - start) * 1000

    _, label, score = decode_predictions(preds, top=1)[0][0]
    text = f"{name}: {label} {score * 100:.0f}% ({ms:.0f} ms)"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, "N = next model, ESC = quit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("Multi-model webcam", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key in (ord("n"), ord("N")):
        active = (active + 1) % len(model_names)

camera.release()
cv2.destroyAllWindows()

# RESULT COMMENTS:
# (fill in after running, compare the ms and scores of each model on the same object)
