# Example 4.15 - WebClassify3.py
# Webcam classification using the Image-Classifiers library (a collection of ready-made models).
# Install first: pip install image-classifiers
# Note: this library was made for older Keras versions. If an import error appears,
# use TensorFlow 2.15 or older.
# Change MODEL_NAME to choose the model. Press ESC to quit.

import cv2
import numpy as np
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from classification_models.keras import Classifiers

# Choices: vgg16, resnet50, mobilenetv2, densenet201, inceptionv3
MODEL_NAME = "vgg16"
SIZE = 299 if MODEL_NAME == "inceptionv3" else 224

Model, preprocess_input = Classifiers.get(MODEL_NAME)
model = Model(input_shape=(SIZE, SIZE, 3), weights="imagenet", classes=1000)
model.summary()

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise SystemExit("Cannot open the webcam.")

while True:
    ok, frame = camera.read()
    if not ok:
        break

    rgb = cv2.cvtColor(cv2.resize(frame, (SIZE, SIZE)), cv2.COLOR_BGR2RGB)
    batch = preprocess_input(np.expand_dims(rgb.astype("float32"), axis=0))

    preds = model.predict(batch, verbose=0)
    _, label, score = decode_predictions(preds, top=1)[0][0]

    cv2.putText(frame, f"{MODEL_NAME}: {label}, {score * 100:.0f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Webcam classification", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

camera.release()
cv2.destroyAllWindows()
