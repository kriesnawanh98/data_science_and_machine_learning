import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

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

# create model of CNN
base_model = tf.keras.applications.nasnet.NASNetMobile(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',  # using transfer learning (or None)
    input_tensor=None,
    pooling=None)

x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

print("Model Summary = \n", model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='binary_crossentropy',
              metrics='accuracy')

history = model.fit(X_train,
                    y_train,
                    epochs=1,
                    validation_data=(X_test, y_test))

pickle.dump(model, open('model_cnn_cancer_classification.pkl',
                        'wb'))  # save the cnn model

# plot the accuracy of the train data and on test/validation data
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy VS Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print("Test loss = ", test_loss)
print("Test accuracy = ", test_acc)
