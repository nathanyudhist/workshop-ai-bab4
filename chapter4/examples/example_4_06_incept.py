# Example 4.6 - Incept.py
# Loads Google Inception V3 (ImageNet weights) and shows its summary.

from tensorflow.keras.applications import InceptionV3

model = InceptionV3(weights="imagenet")
model.summary()
print("Total parameters:", f"{model.count_params():,}")
