# Financial Data Challenge

## Description
This challenge was made within the scope of cooperation between PT. Sharing Vision Indonesia with BRI (PT Bank Rakyat Indonesia, Tbk) with the aim of:
* Opening opportunities for participants from all over Indonesia, from Sabang to Merauke to experience the Challenge of real financial use cases, experiences within the BRI environment.
* Opening up opportunities for participants to get the opportunity to work at BRI. This challenge can help BRI/Sharing Vision to see potential participants to join the Big Data Team within BRI/Sharing Vision.
* Opening opportunities for participants who want to learn by doing.

## Use Case
In this competition, the case that the participants face is churn. Churn is a term in which customers leave the company. If in the world of banking, the customer at a certain period decides to no longer be a customer at one bank and may later become a customer at another bank. Participants will be asked to create a machine learning model for classification that can differentiate between churn and non-churn customers.

The work that participants do must meet the following steps, among others:
1. Exploratory data analysis (EDA)
2. Feature Engineering
3. Modeling
4. Evaluation
5. Prediction Output

## Data

The data consists of two datasets, namely:
* [fin_data_challenge_train_1.csv](findata_challenge_train.csv): 100,000 rows and 126 columns (125 feature columns and 1 target column 'y')
* [fin_data_challenge_test.csv](findata_challenge_test.csv): 25,000 rows and 125 columns (without target column 'y')

**Column Description**<br />
Customer Data: <br />
x0-x124 (125 columns): This is customer data that has been normalized and the column names are kept secret.

Target variable: <br />
y - Will the customer churn? (1 :yes,0 :no)

Note :<br />
There are multiple _Missing Value_ in the data.

## Evaluation
Evaluation is done by measuring the performance of the model conducted by the participants, based on the performance of the AUC.
The submitted file has the following format:
1. Coding script in the form of 'complete_name-submission.ipynb'
2. Output file name (prediction result against test data): 'complete_name-submission.csv' <br /> Columns: 'Id' and 'Predicted'. <br />Please note that file names and column names are case sensitive.
