import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


test_filename="II4035-regresi-test.csv"
train_filename="II4035-regresi-train.csv"

training=np.genfromtxt(train_filename,delimiter=',',skip_header=1)
test    =np.genfromtxt(test_filename,delimiter=',',skip_header=1)

nilai_RMSE = np.array([])
maxdepth_data = np.array([])


for nilai_maxdepth in range(1,3): # banyaknya grafik


    train_X = training[:,1]
    train_y = training[:,2]

    #print("banyak data = ",len(train_X))
    train_X = np.reshape(train_X,(len(train_X),1))

    tree_reg = DecisionTreeRegressor(max_depth= nilai_maxdepth)  
    tree_reg.fit(train_X,train_y)

    X_grid = np.linspace(0,2,100)

    X_grid = X_grid.reshape(len(X_grid),1)

    y_reg = tree_reg.predict(X_grid)
    
    plt.figure('Decision Tree Regression maxdepth= %d' %nilai_maxdepth)
    plt.plot(X_grid, y_reg)
    plt.plot(train_X,train_y,'rx')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Nilai Maxdepth = %d' %nilai_maxdepth)
    

    #RMSE -----------------------------------

    RO = tree_reg.predict(train_X)
    print('RO=',RO)
    RO = np.reshape(RO,(int(len(RO)),1))
    train_y = np.reshape(train_y,(int(len(train_y)),1))
    kw=(RO - train_y)*(RO - train_y)

    RMSE = np.sqrt(sum(kw)/int(len(RO)))


    print('Nilai RMSE =',float(RMSE))
    nilai_RMSE = np.append(nilai_RMSE,RMSE)
    maxdepth_data = np.append(maxdepth_data,nilai_maxdepth)
    
print('nilai rmse',nilai_RMSE)
print('nilai mmaxdept', maxdepth_data)

plt.figure('Decision Tree Regression RMSE')
plt.plot(maxdepth_data,nilai_RMSE,color='green')
plt.xlabel('nilai maxdepth')
plt.ylabel('RMSE')
plt.title('Grafik RMSE vs Nilai Maxdepth')



#--------------------------------------------------------------
test_X = test[:,1]
test_X = np.reshape(test_X,(len(test_X),1))

nilai_maxdepth_prediksi = 2 #<------------ orde yang diprediksi                                   

tree_reg = DecisionTreeRegressor(max_depth= nilai_maxdepth_prediksi)  
tree_reg.fit(train_X,train_y)

y_predict = tree_reg.predict(test_X)

print('y predict=',y_predict)

plt.figure('Decision Tree Regression Predict')
#plt.plot(test_X,y_predict,color='blue')
plt.scatter(test_X,y_predict,color='blue')
plt.xlabel('X Test')
plt.ylabel('Y Predict')
plt.title('Grafik Y Predict vs X test')

plt.show()


#Saving data---------------------------------------------------------------------
y_predict = np.reshape(y_predict,(len(y_predict),))
Y_Predict_data = pd.DataFrame({'Prediksi Y':y_predict})
datatoexcel = pd.ExcelWriter('DecisionTreeRegression.xlsx',engine='xlsxwriter')
Y_Predict_data.to_excel(datatoexcel, sheet_name ='FIT')
datatoexcel.save()
