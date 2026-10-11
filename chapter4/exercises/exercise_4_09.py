# Exercise 4.9
# Task: Modify Example 4.10 so that it loads the VGG19 model instead of VGG16.
# Note: the VGG19 ImageNet weights are downloaded automatically on the first run (about 550 MB).

from tensorflow.keras.applications import VGG19

model = VGG19(weights="imagenet")
model.summary()

print("VGG19 total parameters:", f"{model.count_params():,}")

# RESULT COMMENTS:
# (fill in after running)
# Hint: VGG19 has 3 more convolutional layers than VGG16, so the total is about
# 143.7 million parameters compared to about 138.4 million for VGG16.
