from additional import downgrading_crop_image
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, Sequential, model_from_json
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

input_array = downgrading_crop_image(character="6",
                                     file_name="handyangka_09.jpg")

# load json and create model
json_file = open('model_mnist.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_mnist.h5")
print("Loaded model from disk")

# # evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy',
#                      optimizer='rmsprop',
#                      metrics=['accuracy'])
# score = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

print(input_array)

pred_porba = loaded_model.predict(input_array)
print(pred_porba)
print("Prediction of the image is = ", np.argmax(pred_porba, axis=1)[0])
