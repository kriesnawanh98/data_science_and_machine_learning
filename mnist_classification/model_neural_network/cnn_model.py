import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# read npz file
X_train_ori = np.load('X_train.npz')
X_train_ori = X_train_ori['arr_0']
X_train_ori = X_train_ori / 255.
# X_train_ori = tf.pad(X_train_ori, [[0, 0], [2, 2], [2, 2]]) / 255
X_train_ori = tf.expand_dims(X_train_ori, axis=3, name=None)

y_train_ori = np.load('y_train.npz')
y_train_ori = y_train_ori['arr_0']

X_test = np.load('X_test.npz')
X_test = X_test['arr_0']
X_test = X_test / 255.
# X_test = tf.pad(X_test, [[0, 0], [2, 2], [2, 2]]) / 255
X_test = tf.expand_dims(X_test, axis=3, name=None)

y_test = np.load('y_test.npz')
y_test = y_test['arr_0']

y_test_label = np.load('y_test_label.npz')
y_test_label = y_test_label['arr_0']

# print("shape of X_train_ori data = ", X_train_ori.shape)
# print("shape of y_train_ori data = ", y_train_ori.shape)
# print("shape of X_test data = ", X_test.shape)
# print("shape of y_test data = ", y_test.shape)

X_train = X_train_ori[:50000, :]
y_train = y_train_ori[:50000, :]
X_val = X_train_ori[50000:, :]
y_val = y_train_ori[50000:, :]

print("shape of X_train data = ", X_train.shape)
print("shape of y_train data = ", y_train.shape)
print("shape of X_val data = ", X_val.shape)
print("shape of y_val data = ", y_val.shape)
print("shape of X_test data = ", X_test.shape)
print("shape of y_test data = ", y_test.shape)

# architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train,
                    y_train,
                    epochs=3,
                    batch_size=64,
                    validation_data=(X_val, y_val))

model.evaluate(X_val, y_val)

fig, axs = plt.subplots(2, 1, figsize=(15, 15))
axs[0].plot(history.history['loss'])
axs[0].plot(history.history['val_loss'])
axs[0].title.set_text('Training Loss vs Validation Loss')
axs[0].legend(['Train', 'Val'])
axs[1].plot(history.history['accuracy'])
axs[1].plot(history.history['val_accuracy'])
axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
axs[1].legend(['Train', 'Val'])
plt.show()

# pickle.dump(model, open('model_mnist.pkl', 'wb'))  # save the cnn model

# # serialize model to JSON
# model_json = model.to_json()
# with open("model_mnist.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model_mnist.h5")
# print("Saved model")

y_pred = model.predict(X_test)
print(y_pred)
print("Prediction using test data = ", y_pred.argmax(1))
print("Label of test data = ", y_test_label)

score = 0
for i, val in enumerate(y_pred.argmax(1)):
    if val == y_test_label[i]:
        score += 1

score_percentage = score / (y_test_label.shape[0])
print("accuracy of test data = ", score_percentage)
