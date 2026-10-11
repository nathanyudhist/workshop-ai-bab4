# Example 4.10 - VGG16.py
# Loads the VGG16 model (ImageNet weights) and shows its summary.
# Note: the weights are downloaded automatically on the first run (about 530 MB).

from tensorflow.keras.applications import VGG16

model = VGG16(weights="imagenet")
model.summary()
print("Total parameters:", f"{model.count_params():,}")
