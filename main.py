import pickle
import pandas as pd
import os

#another fuction to define daily_percetange_change
def daily_percentage_change_1(data):
    percentage_changes = [[0 for j in range(2202)] for i in range(2000)]
    for i in range(len(data)):
        for j in range(1, len(data[i])):
            percentage_change = ((data[i]['Close'][j] - data[i]['Close'][j-1] ) / data[i]['Close'][j-1])  * 100
            percentage_changes[i][j-1] = percentage_change
    return percentage_changes

#Calculate the daily percentage changes of the close prices for each stock and return the daily percentage changes 
def daily_percentage_change(data):
    percentage_changes = [df.Close.pct_change() for df in data]
    return percentage_changes

#Divide the daily percentages into three levels. When value is less than -0.05, -1 represents 'Decrease'. If the value is
#greater than 0.05, +1 represents 'Increase'. O.W, the value 0 represents 'no big change'. Retrun 'levels' list
def divide_into_levels(percentage_changes):
    levels = []
    for i in range(len(percentage_changes)):
        stock_levels = []
        for j in range(len(percentage_changes[i])):
            if percentage_changes[i][j] < -0.005:
                stock_levels.append(-1)
            elif percentage_changes[i][j] > 0.005:
                stock_levels.append(1)
            else:
                stock_levels.append(0)
        levels.append(stock_levels)
    return levels



path = os.getcwd()
filename = os.path.join(path, 'training_set.pkl')

with open(filename, 'rb') as f:
    data = pickle.load(f)

#closed_percentage = daily_percentage_change(data)
closed_percentage = daily_percentage_change_1(data)
levels = divide_into_levels(closed_percentage)

print(levels)

