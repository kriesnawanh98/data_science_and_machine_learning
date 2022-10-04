import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import numpy as np
import pickle

# read npz file
X_train = np.load('X_train.npz')
X_train = X_train['arr_0']
X_train = X_train / 255.0

y_train = np.load('y_train.npz')
y_train = y_train['arr_0']

X_test = np.load('X_test.npz')
X_test = X_test['arr_0']
X_test = X_test / 255.0

y_test = np.load('y_test.npz')
y_test = y_test['arr_0']

print("shape of each X data = ", X_train.shape)

model = models.Sequential()
model.add(
    layers.Conv2D(32, (3, 3), activation='relu',
                  input_shape=(224, 224, 3)))  # 32 filters
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

print("Model Summary = \n", model.summary())

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

history = model.fit(X_train,
                    y_train,
                    epochs=10,
                    validation_data=(X_test, y_test))

pickle.dump(model, open('model_cnn_cancer_classification.pkl',
                        'wb'))  # save the cnn model

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print(test_acc)
