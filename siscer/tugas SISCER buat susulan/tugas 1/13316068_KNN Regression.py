import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

test_filename="II4035-regresi-test.csv"
train_filename="II4035-regresi-train.csv"

training=np.genfromtxt(train_filename,delimiter=',',skip_header=1)
test    =np.genfromtxt(test_filename,delimiter=',',skip_header=1)


#print("banyak data = ",len(train_X))

RMSE_data = np.array([])
neighbors_data = np.array([])



for nilai_neighbors in range(1,3): #<---- isi banyaknya looping
    train_X = training[:,1]
    train_y = training[:,2]
    
    '''transpose matrix train_X'''
    train_X = np.reshape(train_X,(len(train_X),1))


    neigh = KNeighborsRegressor(n_neighbors=nilai_neighbors)#UBAH-------<<<<<<--------------------
    neigh.fit(train_X, train_y)


    sampling = np.linspace(0,2,100)

    #sampling = np.reshape(sampling,(100,1))
    sampling = sampling.reshape(100,1)

    y_reg = []

    for i in range(100):
        R = neigh.predict([sampling[i]])
        y_reg.append(R)


    #print(neigh.predict([[sampling]]))
    
    plt.figure('KNN nilai_neighbors = %d' %nilai_neighbors)
    plt.plot(sampling,y_reg,color = 'red')
    plt.scatter(train_X,train_y, color ='blue')
    plt.xlabel('Sumbu X')
    plt.ylabel('Sumbu y')
    plt.title('Nilai Neighbors = %d' %nilai_neighbors)
    

    #RMSE -----------------------------------

    RO = neigh.predict(train_X)
    print('RO=',RO)
    RO = np.reshape(RO,(int(len(RO)),1))
    train_y = np.reshape(train_y,(int(len(train_y)),1))
    kw=(RO - train_y)*(RO - train_y)

    RMSE = np.sqrt(sum(kw)/int(len(RO)))


    print('Nilai RMSE =',float(RMSE))
    RMSE_data = np.append(RMSE_data,RMSE)
    neighbors_data = np.append(neighbors_data,nilai_neighbors)
    
print('RMSE',RMSE_data)
print('neighbors',neighbors_data)


plt.figure('KNN RMSE')
plt.plot(neighbors_data,RMSE_data,color='green')
plt.xlabel('Nilai Neighbors')
plt.ylabel('RMSE')
plt.title('Grafik RMSE vs Nilai Neighbors')

#--------------------------------------------------------------
test_X = test[:,1]
test_X = np.reshape(test_X,(len(test_X),1))

nilai_neighbors_terpilih = 2                            #<------- ISI orde prediksi

neigh = KNeighborsRegressor(n_neighbors=nilai_neighbors_terpilih)
neigh.fit(train_X, train_y)


predict_data_test = neigh.predict(test_X)
print('hasil prediksi = ',predict_data_test)


plt.figure('KNN Regression Predict')
plt.scatter(test_X,predict_data_test,color='blue')
plt.xlabel('X Test')
plt.ylabel('Y Predict')
plt.title('Grafik Y Predict vs X test')

plt.show()


#Saving data---------------------------------------------------------------------
predict_data_test = np.reshape(predict_data_test,(len(predict_data_test),))
Y_Predict_data = pd.DataFrame({'Prediksi Y':predict_data_test})
datatoexcel = pd.ExcelWriter('KNNRegression.xlsx',engine='xlsxwriter')
Y_Predict_data.to_excel(datatoexcel, sheet_name ='data y')
datatoexcel.save()
