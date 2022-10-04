from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import numpy as np

# read npz file
X_train = np.load('X_train.npz')
X_train = X_train['arr_0']

y_train = np.load('y_train.npz')
y_train = y_train['arr_0']

X_test = np.load('X_test.npz')
X_test = X_test['arr_0']

y_test = np.load('y_test.npz')
y_test = y_test['arr_0']

# Training using random forest
n_estimators = 500  # 500 trees
max_depth = 5
rnd_clf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    n_jobs=-1  # using all CPU
)

rnd_clf.fit(X_train, y_train)

y_predict = rnd_clf.predict(X_test)

print("Confusion Matrix = ")
print(confusion_matrix(y_test, y_predict))

print("Accuracy Score = ")
print(accuracy_score(y_test, y_predict))

print("Precision Score = ")
print(precision_score(y_test, y_predict))

print("Recall Score = ")
print(recall_score(y_test, y_predict))