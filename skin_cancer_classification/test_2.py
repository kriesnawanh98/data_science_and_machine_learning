from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

data_x_1 = np.array([0, 1, 1])
print(data_x_1.shape)
data_x_2 = np.array([1, 2, 1])
print(data_x_2.shape)

data_x_3 = np.array([3, 4, 5])
print(data_x_3.shape)

data_x_4 = np.array([3, 4, 5])
print(data_x_4)

data = []
data.append(data_x_1)
data.append(data_x_2)
data.append(data_x_3)
data.append(data_x_4)
data_x = np.array(data)
print(data)
print(data_x.shape)
data_y = np.array([1, 2, 3, 4])
print(data_y.shape)

# x = 4
# n_estimators = 2  # 500 trees
# max_depth = x + 1
# rnd_clf = RandomForestClassifier(n_estimators=n_estimators,
#                                  max_depth=max_depth,
#                                  n_jobs=-1)

# rnd_clf.fit(data_x, data_y)

# from PIL import ImageOps as io
# import numpy as np
# import os
# from PIL import Image

# data_img = Image.open(f"./dataset/train/benign/1799.jpg")
# greyscale_data = io.grayscale(data_img)
# inv_data = io.invert(greyscale_data)  #change from RGB to greyscale
# rsz_data = inv_data.resize((50176, 1))
# numpy_array = np.array(rsz_data.getdata())
# list_data = type(numpy_array)  # Invert/negate the image
# print(numpy_array.shape)