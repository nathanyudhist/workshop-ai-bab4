# Example 4.20 - SimpleRNN3a.py
# Predicts the monthly number of airline passengers (time series) with a SimpleRNN.
# The data is downloaded from the internet, so a connection is needed.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
STEPS = 4  # each prediction uses the 4 previous values

try:
    df = pd.read_csv(URL, usecols=[1])
except Exception as e:
    raise SystemExit(f"Failed to download the data: {e}\nDownload it manually and replace the URL with a file path.")

values = df.values.astype("float32")
plt.plot(values)
plt.title("Passengers per month")
plt.show()

# Scale to 0..1 so that training is more stable
min_v, max_v = values.min(), values.max()
scaled = (values - min_v) / (max_v - min_v)

split = int(len(scaled) * 0.8)
train, test = scaled[:split, 0], scaled[split:, 0]


def make_windows(series, steps):
    X, y = [], []
    for i in range(len(series) - steps):
        X.append(series[i:i + steps])
        y.append(series[i + steps])
    return np.array(X), np.array(y)


X_train, y_train = make_windows(train, STEPS)
X_test, y_test = make_windows(test, STEPS)

# Shape for the RNN: (samples, time steps, features)
X_train = X_train.reshape(-1, 1, STEPS)
X_test = X_test.reshape(-1, 1, STEPS)

model = keras.Sequential([
    layers.Input(shape=(1, STEPS)),
    layers.SimpleRNN(32, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1),
])
model.compile(loss="mean_squared_error", optimizer="rmsprop")
model.summary()

model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=2)

pred_train = model.predict(X_train, verbose=0).ravel()
pred_test = model.predict(X_test, verbose=0).ravel()


# Convert back to the original scale to compute the error
def restore(a):
    return a * (max_v - min_v) + min_v


rmse_train = np.sqrt(np.mean((restore(pred_train) - restore(y_train)) ** 2))
rmse_test = np.sqrt(np.mean((restore(pred_test) - restore(y_test)) ** 2))
print(f"Train RMSE: {rmse_train:.2f}  Test RMSE: {rmse_test:.2f}")

# Result plot
plt.figure(figsize=(10, 4))
plt.plot(range(len(scaled)), restore(scaled[:, 0]), label="Actual data")
plt.plot(range(STEPS, STEPS + len(pred_train)), restore(pred_train), label="Train prediction")
plt.plot(range(split + STEPS, split + STEPS + len(pred_test)), restore(pred_test), label="Test prediction")
plt.legend()
plt.show()
