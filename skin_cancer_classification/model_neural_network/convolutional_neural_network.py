import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # hanya diperlukan untuk disable GPU

import tensorflow as tf
import tensorflow.keras as keras  # pakai Keras dari tensorflow
import os
from tensorflow.keras.layers import Flatten, Dense, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model
import numpy as np
from pathlib import Path
import shutil

#prepare model
img_width = 224  #ukuran lebar gambar
img_height = 224  # ukuran tinggi gambar

cnn_notop = keras.applications.nasnet.NASNetMobile(
    input_shape=(img_width, img_height, 3),
    include_top=False
    #                                                 , weights='imagenet' # menggunakan transfer learning
    ,
    weights=None  # tanpa transfer learning
    ,
    input_tensor=None,
    pooling=None)

# In[8]:

#merancang cnn
x = cnn_notop.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
predictions = Dense(12, activation='softmax')(x)
the_model = Model(cnn_notop.input, predictions)

#training
learning_rate = 0.0001
logfile = session + '-train' + '.log'
batch_size = 32
nbr_epochs = 2  # jumlah nilai epoch = 2
print("training  directory: " + train_dir)
print("valication directory: " + valid_dir)
optimizer = SGD(lr=learning_rate, momentum=0.9, decay=0.0, nesterov=True)
the_model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
csv_logger = CSVLogger(logfile, append=True)
early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=10,
                               verbose=1,
                               mode='auto')
best_model_filename = session + '-weights.{epoch:02d}-{val_loss:.2f}.h5'
best_model = ModelCheckpoint(best_model_filename,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True)

# fungsi fit untuk model
the_model.fit(x=train_generator,
              epochs=nbr_epochs,
              verbose=1,
              validation_data=validation_generator,
              callbacks=[best_model, csv_logger, early_stopping])

print('Begin to predict for testing data ...')
predictions = predict_model.predict(x=test_generator,
                                    steps=nbr_test_samples / batch_size,
                                    verbose=1)
np.savetxt(
    session + '-predictions.txt',
    predictions)  # store prediction matrix, for later analysis if necessary
