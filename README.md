I have been granted access to historical stock market information and my objective involves utilizing this data to train and evaluate different machine learning models designed for predicting stock prices. To achieve this, I've been provided with three distinct datasets. These datasets consist of one training set and two testing sets, which can be loaded into the program using the pickle.load() function.

The training set is composed of a collection of 2000 pandas data frames. Each data frame corresponds to the historical trading data of an individual stock. In total, the dataset encompasses 2000 unique stocks. Each of these Pandas data frames contains a series of trading records that span consecutive time points. It's important to note that all numerical values within the dataset have undergone a normalization process. (fill the missing data and remove noisy data)

Task Description: 
In this project, I will train a model to predict future stock prices based on past trading records. This is a supervised learning problem, where the input features consist of a matrix with 100 rows and m (m>=100) columns, and the label is a 3-dimensional vector. We will provide further details on this later on.

I have studied stock indicators to see which indicators would help to increase the possibility of prediction. There are more than 100 features for each stock.

Task3,4
Implemented 10 different Neural net models. Every time when it gets higher accuracy, the best model is saved based on the best accuracy for each model.

Task5
Utilized the methods of blending, Voting, and Adaboost to boost model's performance of my 10 models
(Printed accuracy, precision, and positive prediction for training and validation set).

Task6
Chose the best model from task5, and evaluated the performance of my final model on the three testing sets: Vpting, Blending, and Adaboost
