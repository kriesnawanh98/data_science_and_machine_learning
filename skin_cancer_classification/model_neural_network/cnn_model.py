import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.utils import shuffle

# read train data
X_train_ori = np.load('X_train.npz')
X_train_ori = X_train_ori['arr_0']
X_train_ori = X_train_ori / 255.0

y_train_ori = np.load('y_train.npz')
y_train_ori = y_train_ori['arr_0']

X_train_ori, y_train_ori = shuffle(X_train_ori, y_train_ori)
# split data train_ori into train and validation
X_train = X_train_ori[:2000]
y_train = y_train_ori[:2000]

X_val = X_train_ori[2000:]
y_val = y_train_ori[2000:]

# read test data
X_test = np.load('X_test.npz')
X_test = X_test['arr_0']
X_test = X_test / 255.0

y_test = np.load('y_test.npz')
y_test = y_test['arr_0']

# check size of train, val and test data
print("data X_train = ", X_train.shape)
print("data y_train = ", y_train.shape)
print("data X_val = ", X_val.shape)
print("data y_val = ", y_val.shape)
print("data X_test = ", X_test.shape)
print("data y_test = ", y_test.shape)

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
              loss='categorical_crossentropy',
              metrics='accuracy')

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# pickle.dump(model, open('model_cnn_cancer_classification.pkl',
#                         'wb'))  # save the cnn model

# plot the accuracy of the train data and on test/validation data
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy VS Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=2)
print("Validation loss = ", val_loss)
print("Validation accuracy = ", val_acc)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print("Test loss = ", test_loss)
print("Test accuracy = ", test_acc)

y_pred = model.predict(X_test)
score = 0
y_test_label = y_test.argmax(1)
for i, val in enumerate(y_pred.argmax(1)):
    if val == y_test_label[i]:
        score += 1

score_percentage = score / (y_test_label.shape[0])
print("accuracy of test data = ", score_percentage)