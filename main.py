import pickle
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


#Calculate the daily percentage changes of the close prices for each stock and return the daily percentage changes 
def daily_percentage_change_for_close(data):
    percentage_changes = [df.Close.pct_change()*100 for df in data]
    return percentage_changes

#Divide the daily percentages into three levels. When value is less than -0.05, -1 represents 'Decrease'. If the value is
#greater than 0.05, +1 represents 'Increase'. O.W, the value 0 represents 'no big change'. Retrun 'levels' list
def divide_into_levels(percentage_changes, thresholds):
    levels = []
    for i in range(len(percentage_changes)):
        stock_levels = []
        for j in range(len(percentage_changes[i])):
            if percentage_changes[i][j] < thresholds[i][0]:
                stock_levels.append(-1)
            elif percentage_changes[i][j] >thresholds[i][1]:
                stock_levels.append(1)
            else:
                stock_levels.append(0)
        levels.append(stock_levels)
    return levels

#print the two threshold used to divde data into three levels
#add the number of data points for each level and print them
def threshold_number_of_data_points(levels, thresholds):

    stock = 1
    
    for i in range(len(levels)):
        print("[stock] :", stock)
        increase = sum([1 for level in levels[i] if level == 1])
        decrease = sum([1 for level in levels[i] if level == -1])
        no_change = sum([1 for level in levels[i] if level == 0])
        print("Threshold 1:", thresholds[i][0])
        print("Threshold 2:", thresholds[i][1])
        print("Number of increases:", increase)
        print("Number of decreases:", decrease)
        print("Number of no changes:", no_change)
        stock += 1


#Calculate the median and standard deviation of the daily percentage changes
def find_thresholds(closed_percentage):
    sorted_closed_percentage = [df.sort_values() for df in closed_percentage]
    thresholds = []
    
    for df in sorted_closed_percentage:
        threshold1 = df.quantile(33.3/100)
        threshold2 = df.quantile(66.6/100)
        thresholds.append((threshold1, threshold2))
    return thresholds

#show distribution of data
def show_distribution(data):

    plt.hist(data, bins=50, range=(-1,1))
    plt.show()

#main function
def main():
    path = os.getcwd()
    filename = os.path.join(path, 'training_set.pkl')

    with open(filename, 'rb') as f:
        data = pickle.load(f)
   
    closed_percentage = daily_percentage_change_for_close(data)
    thresholds = find_thresholds(closed_percentage)
  
    levels = divide_into_levels(closed_percentage, thresholds)
    threshold_number_of_data_points(levels, thresholds)

main()



