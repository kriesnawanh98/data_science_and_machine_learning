import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model



test_filename="II4035-regresi-test.csv"
train_filename="II4035-regresi-train.csv"

training=np.genfromtxt(train_filename,delimiter=',',skip_header=1)
test    =np.genfromtxt(test_filename,delimiter=',',skip_header=1)

train_X = training[:,1]
train_y = training[:,2]
print("banyak data = ",len(train_X))

data_RMSE = np.array([[]])
RMSE_X = np.array([])


#polynomial Regression orde  n 
data_RMSE = np.array([])


for indeks_orde in range(1,3):           #<----------input orde
    train_X = training[:,1]
    train_y = training[:,2]

    '''transpose matrix train_X'''
    train_X = np.reshape(train_X,(len(train_X),1))

    
    
    def fungsi_pangkat(input_nilai,pangkat_berapa):
        hasil = bilangan = input_nilai
        for c in range(pangkat_berapa-1):
            hasil = hasil*bilangan
        return(hasil)
   
    
    orde = indeks_orde      
    poly_features = preprocessing.PolynomialFeatures(degree=orde, include_bias = False)

    X_poly = poly_features.fit_transform(train_X) #matrix tegak

    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(X_poly, train_y) #y matrix mendatar

    intercept = lin_reg.intercept_
    coef = lin_reg.coef_

    print('intercept',lin_reg.intercept_,'coef=',lin_reg.coef_)

    sampling = np.linspace(0,2,100)
    hasilnya = 0

    for i in range(orde):
        hasilnya = hasilnya + coef[i]*(fungsi_pangkat(sampling,i+1))

    W = hasilnya + float(intercept)     


    
    plt.figure('Polynomial Regression orde= %d' %indeks_orde)
    plt.plot(sampling,W,'r-')
    #plt.plot(rang,data_linear_y,'m-')
    plt.plot(train_X,train_y,'bx')
    plt.xlabel('sumbu X')
    plt.ylabel('sumbu y')
    plt.title('Orde ke - %d' %indeks_orde)
 


    #nilai RMSE (Root Mean Square Error)

    hasilnya = 0
    for i in range(orde):
        hasilnya = hasilnya + coef[i]*(fungsi_pangkat(train_X,i+1))
    Wo = hasilnya + float(intercept)

    print('jml data:',len(Wo))


    train_y = np.reshape(train_y,(int(len(train_y)),1))
    kw=(Wo - train_y)*(Wo - train_y)

    RMSE = np.sqrt(sum(kw)/int(len(Wo)))
    print('Nilai RMSE =',RMSE)
    data_RMSE = np.append(data_RMSE,[RMSE])
    RMSE_X = np.append(RMSE_X,orde)
    
    
    
print('data RMSE =',data_RMSE)
print('RMSE_X =',RMSE_X)

plt.figure('Polynomial Regression RMSE')
plt.plot(RMSE_X,data_RMSE,color='green')
plt.xlabel('orde')
plt.ylabel('RMSE')
plt.title('Nilai RMSE VS orde')



#--------------------------------------------------------------
train_X = training[:,1]
train_y = training[:,2]

#transpose matrix train_X
train_X = np.reshape(train_X,(len(train_X),1))



test_X = test[:,1]
test_X = np.reshape(test_X,(len(test_X),1))


def fungsi_pangkat(input_nilai,pangkat_berapa):
    hasil = bilangan = input_nilai
    for c in range(pangkat_berapa-1):
        hasil = hasil*bilangan
    return(hasil)
   
    
orde_prediksi = 2                           #<------------------input orde prediksi     
poly_features = preprocessing.PolynomialFeatures(degree=orde_prediksi, include_bias = False)

X_poly = poly_features.fit_transform(train_X) #matrix tegak

lin_reg = linear_model.LinearRegression()
lin_reg.fit(X_poly, train_y) #y matrix mendatar

intercept = lin_reg.intercept_
coef = lin_reg.coef_

print('intercept',lin_reg.intercept_,'coef=',lin_reg.coef_)

sampling = test_X
hasilnya = 0

for i in range(orde_prediksi):
    hasilnya = hasilnya + coef[i]*(fungsi_pangkat(sampling,i+1))

W = hasilnya + float(intercept)     


#W = coef[4]*sampling*sampling*sampling*sampling*sampling+float(coef[3])*sampling*sampling*sampling*sampling+float(coef[2])*sampling*sampling*sampling +float(coef[1])*sampling*sampling + float(coef[0])*sampling +float(intercept)
#DIUBAH

plt.figure('Hasil Prediksi orde= %d' %orde_prediksi)
plt.scatter(sampling,W,color='red')
#plt.plot(rang,data_linear_y,'m-')
plt.xlabel('X_Test')
plt.ylabel('Y_Predict')
plt.title('Hasil Prediksi dengan Orde ke - %d' %orde_prediksi)
plt.show()


#Saving data---------------------------------------------------------------------
W = np.reshape(W,(len(W),))
Y_Predict_data = pd.DataFrame({'Prediksi Y':W})
datatoexcel = pd.ExcelWriter('PolynomialRegression.xlsx',engine='xlsxwriter')
Y_Predict_data.to_excel(datatoexcel, sheet_name ='data y')
datatoexcel.save()
