# Example 4.14 - WebcamClassify2.py
# Webcam classification with a choice of model: VGG16, ResNet50, MobileNet, or InceptionV3.
# Change MODEL_NAME to choose the model. Press ESC to quit.

import cv2
import numpy as np
from tensorflow.keras.applications import vgg16, resnet50, mobilenet, inception_v3
from tensorflow.keras.applications.imagenet_utils import decode_predictions

MODEL_NAME = "resnet50"  # choices: vgg16, resnet50, mobilenet, inceptionv3

MODEL_OPTIONS = {
    "vgg16": (vgg16.VGG16, vgg16, 224),
    "resnet50": (resnet50.ResNet50, resnet50, 224),
    "mobilenet": (mobilenet.MobileNet, mobilenet, 224),
    "inceptionv3": (inception_v3.InceptionV3, inception_v3, 299),
}

builder, module, size = MODEL_OPTIONS[MODEL_NAME]
model = builder(weights="imagenet")
model.summary()

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise SystemExit("Cannot open the webcam.")

while True:
    ok, frame = camera.read()
    if not ok:
        break

    rgb = cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2RGB)
    batch = module.preprocess_input(np.expand_dims(rgb.astype("float32"), axis=0))

    preds = model.predict(batch, verbose=0)
    _, label, score = decode_predictions(preds, top=1)[0][0]

    cv2.putText(frame, f"{label}, {score * 100:.0f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.imshow("Webcam classification", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

camera.release()
cv2.destroyAllWindows()
