import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR

test_filename = "II4035-regresi-test.csv"
train_filename = "II4035-regresi-train.csv"

training = np.genfromtxt(train_filename, delimiter=',', skip_header=1)
test = np.genfromtxt(test_filename, delimiter=',', skip_header=1)

train_X = training[:, 1]
train_y = training[:, 2]

print("banyak data = ", len(train_X))
'''transpose matrix train_X'''
train_X = np.reshape(train_X, (len(train_X), 1))

#polynomial
from sklearn.svm import SVR

data_RMSE = np.array([])
data_orde = np.array([])

for indeks_orde in range(1, 10):  #<-------------------input orde
    orde = indeks_orde
    reg_poly = SVR(kernel='poly',
                   C=100,
                   gamma='auto',
                   degree=orde,
                   epsilon=0.1,
                   coef0=1)

    reg_poly.fit(train_X, train_y)

    sample = np.linspace(0, 2, 100)
    sample = np.reshape(sample, (100, 1))

    ty = reg_poly.predict(sample)
    plt.figure('SVM orde= %d' % indeks_orde)
    plt.plot(sample, ty, color="red")
    plt.scatter(train_X, train_y, color='blue')
    plt.xlabel('sumbu x')
    plt.ylabel('sumbu y')
    plt.title('orde ke - %d' % indeks_orde)

    # RMSE --------------------------------------------

    Wo = reg_poly.predict(train_X)
    Wo = np.reshape(Wo, (int(len(Wo)), 1))

    train_y = np.reshape(train_y, (int(len(train_y)), 1))
    kw = (Wo - train_y) * (Wo - train_y)

    RMSE = np.sqrt(sum(kw) / int(len(Wo)))
    print('Nilai RMSE =', float(RMSE))
    data_RMSE = np.append(data_RMSE, RMSE)
    data_orde = np.append(data_orde, indeks_orde)

print('RMSE data = ', data_RMSE)
print('ORDE data = ', data_orde)
plt.figure('SVM RMSE')
plt.plot(data_orde, data_RMSE, color='green')
plt.xlabel('orde')
plt.ylabel('RMSE')
plt.title('RMSE VS orde')

#--------------------------------------------------------------
test_X = test[:, 1]

orde = 2  #<---------------------------------------input orde prediksi
reg_poly = SVR(kernel='poly',
               C=100,
               gamma='auto',
               degree=orde,
               epsilon=0.1,
               coef0=1)

reg_poly.fit(train_X, train_y)

sample = test_X
sample = np.reshape(sample, (len(sample), 1))

ty = reg_poly.predict(sample)

plt.figure('SVM prediksi orde= %d' % orde)
plt.scatter(sample, ty, color="red")
plt.xlabel('X_Test')
plt.ylabel('Y_Predict')
plt.title('Prediksi orde ke - %d' % orde)
plt.show()

#Saving data---------------------------------------------------------------------
ty = np.reshape(ty, (len(ty),))
Y_Predict_data = pd.DataFrame({'Prediksi Y': ty})
datatoexcel = pd.ExcelWriter('SVMregression.xlsx', engine='xlsxwriter')
Y_Predict_data.to_excel(datatoexcel, sheet_name='data y')
datatoexcel.save()
