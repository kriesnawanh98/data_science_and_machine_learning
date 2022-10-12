from keras.datasets import mnist
import pickle
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, MaxPool2D
from tensorflow.keras.optimizers import SGD

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train = X_train / 255.
# X_test = X_test / 255.

print("Shape of X_train = ", X_train.shape)
print("Shape of X_test = ", X_test.shape)

print("Shape of y_train = ", y_train.shape)
print("Shape of y_train = ", y_test.shape)

print("List of label = ", np.unique(y_train))
print("size of unique label = ", np.unique(y_train).shape)

unique_label_size = np.unique(y_train).shape
n_label = unique_label_size[0]
print(n_label)

# print(X_train[0])

# plt.imshow(X_train[0])
# plt.show()

# X_train = np.expand_dims(X_train, -1)
# X_test = np.expand_dims(X_test, -1)
# print(X_train.shape)
# print(X_train[0].shape)

y_train = keras.utils.to_categorical(y_train, n_label)
y_test = keras.utils.to_categorical(y_test, n_label)

print(y_train.shape)

np.savez_compressed('X_train.npz', X_train)
np.savez_compressed('y_train.npz', y_train)
np.savez_compressed('X_test.npz', X_test)
np.savez_compressed('y_test.npz', y_test)
