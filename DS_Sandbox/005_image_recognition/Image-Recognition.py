import keras as keras
import tensorflow as ten

print(keras.__version__)
print(ten.__version__)

import gzip
import numpy as np

def open_images(filename):
    with gzip.open(filename, "rb") as file:
        data = file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=16) \
            .reshape(-1, 28, 28) \
            .astype(np.float32)

def open_labels(filename):
    with gzip.open(filename, "rb") as file:
        data = file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=8)

X_train = open_images("data/fashion/train-images-idx3-ubyte.gz")
y_train = open_labels("data/fashion/train-labels-idx1-ubyte.gz")

print("1. Bild")
print(X_train[0])

y_train = y_train == 0 #Bild zeigt T-Shirt

# print(X_train)
# print(X_train.shape)

import matplotlib.pyplot as plt

# print(y_train)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
#hiddenlayer
model.add(Dense(100, activation="sigmoid", input_shape=(784,)))
#outputlayer
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="sgd", loss="binary_crossentropy")

a = X_train.reshape(60000,784)

model.fit(a, y_train, epochs=10, batch_size=1000)

print(y_train[0])
plt.imshow(X_train[0], cmap='gray_r')
plt.show()
print(model.predict(X_train[0].reshape(1,784)))

print(y_train[1])
plt.imshow(X_train[1], cmap='gray_r')
plt.show()
print(model.predict(X_train[1].reshape(1,784)))
