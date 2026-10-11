# Example 4.3 - MPL2.py
# Keras MLP for breast cancer classification (built-in Scikit-Learn dataset).
# Model: input, two dense layers with 10 neurons each, and one output neuron.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=7
)

# Standardize so that training is more stable
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(10, activation="relu"),
    layers.Dense(10, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, batch_size=50, validation_split=0.2, epochs=100, verbose=1)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("loss    :", loss)
print("accuracy:", acc)
