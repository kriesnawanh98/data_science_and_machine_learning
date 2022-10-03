from additional import convert_image_to_array
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

X_benign_train, y_benign_train = convert_image_to_array(
    path="./dataset/train/benign/", label="benign")
X_malignant_train, y_malignant_train = convert_image_to_array(
    path="./dataset/train/malignant/", label="malignant")

X_train = np.concatenate((X_benign_train, X_malignant_train), axis=0)
y_train = np.concatenate((y_benign_train, y_malignant_train), axis=0)

X_benign_test, y_benign_test = convert_image_to_array(
    path="./dataset/test/benign/", label="benign")
X_malignant_test, y_malignant_test = convert_image_to_array(
    path="./dataset/test/malignant/", label="malignant")

X_test = np.concatenate((X_benign_test, X_malignant_test), axis=0)
y_test = np.concatenate((y_benign_test, y_malignant_test), axis=0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

np.savez_compressed('X_train.npz', X_train)
np.savez_compressed('y_train.npz', y_train)
np.savez_compressed('X_test.npz', X_test)
np.savez_compressed('y_test.npz', y_test)

# x = 4
# n_estimators = 2  # 500 trees
# max_depth = x + 1
# rnd_clf = RandomForestClassifier(n_estimators=n_estimators,
#                                  max_depth=max_depth,
#                                  n_jobs=-1)

# rnd_clf.fit(X_train, y_train)

# print("Predict = ", rnd_clf.predict([X_test[0]]))
# print("actualnya = ", y_test[0])
