# Task One - Regression

## 1. Polynomial Regression
Polynomial Regression is a combination of linear regression which is formed by
independent variable x and dependent variable y which can be modeled with polynomial order
. Each order that is applied will have a Root Mean Square Error (RMSE) value that is
vary. Therefore, the smallest RMSE value is needed, but this must be balanced
by the proportional graph so that there is no overfit.

### 1.1 Graph of Best Fit

The best model uses order 10

<img src="image/polynomial/Polynomial_Regression_orde=_10 (BEST FIT).png">

<img src="image/polynomial/Hasil_Prediksi_orde=_10 (best).png">

<img src="image/polynomial/Polynomial_Regression_RMSE.png">


## 2. Support Vector Machine (SVM)
Support Vector Machine (SVM) regression is a nonparametric method where
based on kernel functions. SVM regression can be used for both linear and regression
polynomials. This method has a hyperparameter variable to limit errors.
When the value of gets bigger, more data will be used to construct
regression line but this will give a larger error value.

### 2.1 Graph of Best Fit

The best model uses order 8

<img src="image/SVM/SVM_orde=_8 (BEST FIT).png">

<img src="image/SVM/SVM_prediksi_orde=_8 (best).png">

<img src="image/SVM/SVM_RMSE.png">



## 3. KNN Regression
K nearest neighbors (KNN) regression is a method to predict a target
numerical data based on the same measurement. For a high value of K will
provide higher regression precision.

### 3.1 Graph of Best Fit
The best model uses neighbors = 3
<img src="image/KNN/KNN_nilai_neighbors_=_3 (BEST FIT).png">

<img src="image/KNN/KNN_Regression_Predict (best).png">

<img src="image/KNN/KNN_RMSE.png">


## 4. Decision Tree Regression
Decision Tree Regression is a method for dividing a data set into several groups
part. Each data set will be represented by its average value. Amount of data sharing
expressed by the variable max_depth in this regression.

### 4.1 Graph of Best Fit

The best model uses max_depth = 7

<img src="image/decision_tree/7. Decision_Tree_Regression_maxdepth=_7 (BEST FIT).png">

<img src="image/decision_tree/Decision_Tree_Regression_Predict (best).png">

<img src="image/decision_tree/Decision_Tree_Regression_RMSE.png">