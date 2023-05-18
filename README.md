I have been provided with historical stock market data and my task is to use the data to train and test various machine learning models for predicting stock prices. Specifically, I have been given three datasets, namely one training set and two testing sets, which can be loaded using the pickle.load() function.The training set is a list of 2000 pandas dataframes, each representing the historical trading data of a single stock. There are a total of 2000 stocks in the dataset. Each pandas dataframe contains 2202 trading records for consecutive time points, each record including 5 features, namely “Open”, “Low”, “High”, “Close”, and “Volume”. Note that all the numbers in the dataset have been normalized.

Task Description: 
In this project, I will be working on training a model to predict the future stock prices based on the past 100 trading records. This is a supervised learning problem, where the input features consist of a matrix with 100 rows and m (m>=100) columns, and the label is a 3-dimensional vector. We will provide further details on this later on.

there are more than 100 features for each stocks.

Task3,4
Implemented 10 different Nueral net models. Everytime when it gets higher accuracy, the bestmodel is saved based on the best accuracy for each model.

Task5
Utilized the methods of blending, Voting, and Adaboost to boost model's performance of my 10 models
(Printed accuracy, precision, and positive prediction for training and validation set).

Task6
Choosed the best model from task5, and evaluate the perforamnce of my final model on the three testing sets.
