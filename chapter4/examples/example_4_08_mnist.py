# Example 4.8 - MNIST.py
# Loads the MNIST handwritten digit dataset and shows the first four images.

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Training data:", x_train.shape, " Test data:", x_test.shape)

plt.figure(figsize=(8, 2.5))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")
plt.show()
