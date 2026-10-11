# Example 4.13 - WebcamClassify.py
# Live image classification from the webcam using VGG16.
# Press ESC to quit.

import cv2
import numpy as np
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.imagenet_utils import decode_predictions

SIZE = 224
model = vgg16.VGG16(weights="imagenet")
model.summary()

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise SystemExit("Cannot open the webcam.")

while True:
    ok, frame = camera.read()
    if not ok:
        break

    # OpenCV gives BGR images, but ImageNet models need RGB
    rgb = cv2.cvtColor(cv2.resize(frame, (SIZE, SIZE)), cv2.COLOR_BGR2RGB)
    batch = vgg16.preprocess_input(np.expand_dims(rgb.astype("float32"), axis=0))

    preds = model.predict(batch, verbose=0)
    _, label, score = decode_predictions(preds, top=1)[0][0]

    text = f"VGG16: {label} {score * 100:.0f}%"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.imshow("Webcam classification", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

camera.release()
cv2.destroyAllWindows()
