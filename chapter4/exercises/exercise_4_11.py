# Exercise 4.11
# Task: Modify the VGG16 webcam program (Example 4.13) so that it uses VGG19.
# Compare the performance.
#
# To compare: run this file, then change "vgg19" to "vgg16" in the import and in the model line,
# and compare the ms values and scores shown on screen.
# Press ESC to quit.

import time
import cv2
import numpy as np
from tensorflow.keras.applications import vgg19
from tensorflow.keras.applications.imagenet_utils import decode_predictions

SIZE = 224
model = vgg19.VGG19(weights="imagenet")

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise SystemExit("Cannot open the webcam.")

while True:
    ok, frame = camera.read()
    if not ok:
        break

    # OpenCV gives BGR images, but ImageNet models need RGB
    rgb = cv2.cvtColor(cv2.resize(frame, (SIZE, SIZE)), cv2.COLOR_BGR2RGB)
    batch = vgg19.preprocess_input(np.expand_dims(rgb.astype("float32"), axis=0))

    start = time.perf_counter()
    preds = model.predict(batch, verbose=0)
    ms = (time.perf_counter() - start) * 1000

    _, name, score = decode_predictions(preds, top=1)[0][0]
    text = f"VGG19: {name} {score * 100:.0f}% ({ms:.0f} ms)"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow("VGG19 webcam", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

camera.release()
cv2.destroyAllWindows()

# RESULT COMMENTS:
# (fill in after running)
