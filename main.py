import pickle
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt



#-------------------------------------------------------------------------------Task1------------------------------------------------------------------------------------------------
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
            if percentage_changes[i][j] < thresholds[0][0]:
                stock_levels.append(-1)
            elif percentage_changes[i][j] >thresholds[0][1]:
                stock_levels.append(1)
            else:
                stock_levels.append(0)
        levels.append(stock_levels)
    return levels

#print the two threshold used to divde data into three levels
#add the number of data points for each level and print them
def threshold_number_of_data_points(levels, thresholds):

    increase = 0
    decrease = 0
    no_change = 0

    for i in range(len(levels)):
        increase += sum([1 for level in levels[i] if level == 1])
        decrease += sum([1 for level in levels[i] if level == -1])
        no_change += sum([1 for level in levels[i] if level == 0])

    print("Threshold 1:", thresholds[0][0])
    print("Threshold 2:", thresholds[0][1])
    print("Number of increases:", increase)
    print("Number of decreases:", decrease)
    print("Number of no big changes:", no_change)
       


#To get the two threshold, we use 33th and 66th percentile.
#return list of thresholds   
def find_thresholds(closed_percentage):

    #combine 2000X2202 data int one dimesional data
    combine_into_one = np.array(closed_percentage).ravel()
    sorted_closed_percentage = np.sort(combine_into_one)
    thresholds = []
    threshold1 = np.nanpercentile(sorted_closed_percentage, 33.3)
    threshold2 = np.nanpercentile(sorted_closed_percentage, 66.6)
    thresholds.append((threshold1, threshold2))

    return thresholds
   

#show distribution of data
def show_distribution(data):
    plt.hist(data, bins=50, range=(-1,1))
    plt.show()

#show plot for closing price
def visualize_price(data):
    plt.figure(figsize = (16,8))
    plt.title('The closing price of stock')
    plt.plot(data['Close'])
    plt.xlabel('number', fontsize = 20)
    plt.ylabel('Close Price', fontsize = 20)
    plt.show()

#--------------------------------------------------------------------------------Task2-----------------------------------------------------------------------------------------------------

#calculate a daily return which is calculated by substracting the opening price from the closing price
def daily_return(data):
    daily_ret= [df.Close - df.Open for df in data]
    return daily_ret

#calculate a moving average of closing prices with given data and given days(e.g. The window_size argument specifies the number of days over which the average is taken.)
def moving_average(data, window_size):
    moving_averages = []
    for df in data:
        close_prices = df['Close']
        moving_average = close_prices.rolling(window_size).mean()
        moving_averages.append(moving_average)
    return moving_averages

#calculates upper and lower Bollinger Bands, using moving_averge and two standard deviation and return lists of upperbnads and lower bands
def bollinger_bands(data, window_size, k):
    upper_bands = []
    lower_bands = []
    #get moving average
    moving_averages = moving_average(data, window_size)

    for df, moving in zip(data, moving_averages):
        close_prices = df['Close']
        std_dev = close_prices.rolling(window_size).std()
        upper_band = moving + k * std_dev
        lower_band = moving - k * std_dev
        upper_bands.append(upper_band)
        lower_bands.append(lower_band)
    return upper_bands, lower_bands

#Expoenetial Moving Average that gives more weight to recent prices and return the list of Exponential Moving Average
def ema(data, window):
    ema = []
    for i in range(len(data)):
        df = data[i]
        weight = 2/(window + 1)
        exponetial = df.Close.ewm(span=window, adjust=False).mean()
        ema.append(exponetial)
    return ema


# Calculate weighted_moving_average that assigns higher weight to more recent values and lower weight to older values
# need to fix
def weighted_moving_average(data, weighting_factor):
    wma = []
    for df in data:
        stock_wma = 0
        for j, value in enumerate(df['Close']):
            weight = weighting_factor - j
            stock_wma += value * weight
        stock_wma /= sum(range(1, int(weighting_factor)+1))
        wma.append(stock_wma)
    return wma

# Calculate Momentum that measures the strength fo the price trend in an asset and return the list of mementum
def momentum(data, n):
    momentum = []
    for df in data:
        stock_momentum = df.Close.diff(n)
        momentum.append(stock_momentum)
    return momentum

#Rate of Change
#Measures the percentage change the current closing price and the closing price n-time periods ago and return the list of Rate of Change
def ROC(data, n):
    roc = []
    for i in range(len(data)):
        stock_roc = (data[i]['Close'] - data[i]['Close'].shift(n)) / data[i]['Close'].shift(n) * 100
        roc.append(stock_roc)
    return roc

#Relative Strength Index
#Momentum indicator that compares the magnitude of recent gains to recent losses to determine overbought or oversold conditions in an asset.
#return the list of RSI 
def RSI(data, n):

    results = []
    for df in data:
        change = df['Close'].diff()
        gains = change.where(change > 0, 0)
        losses = -change.where(change < 0, 0)
        avg_gain = gains.rolling(n).mean()
        avg_loss = losses.rolling(n).mean()
        RS = avg_gain / avg_loss
        RSI = 100 - (100 / (1 + RS))
        results.append(RSI)
    return results


#Stochastic Oscillator
#momentum indicator that compares a security's close price to its price range over a specified number of periods
def stochastic_oscillator(data, n):
    results = []
    for df in data:
        close = df['Close']
        low = df['Low'].rolling(n).min()
        high = df['High'].rolling(n).max()
        stochastic = 100 * (close - low) / (high - low)
        results.append(stochastic)
    return results

#Calculate Moving Average Convergence Divergence
#reutrn MACD, MACD histogram, signal line
def MACD(data, fast_period=12, slow_period=26, signal_period=9):
    macd = []
    signal = []
    histogram = []
    for df in data:
        fast_ma = df['Close'].rolling(window=fast_period).mean()
        slow_ma = df['Close'].rolling(window=slow_period).mean()
        macd.append(fast_ma - slow_ma)
        signal_line = macd[-1].rolling(window=signal_period).mean()
        signal.append(signal_line)
        histogram.append(macd[-1] - signal_line)
    return macd, signal, histogram


#-------------------------------------------------------------------------------Mainfunction-----------------------------------------------------------------------------------
def main():
    path = os.getcwd()
    filename = os.path.join(path, 'testing_set2.pkl')

    with open(filename, 'rb') as f:
        data = pickle.load(f)
    

    #Task1
    #closed_percentage = daily_percentage_change_for_close(data)
    #thresholds = find_thresholds(closed_percentage)
    #levels = divide_into_levels(closed_percentage, thresholds)
    #threshold_number_of_data_points(levels, thresholds)

    #Taks2
    #to print specific row data
    #print(data[0]['Close'].iloc[0:40])


    #daily_return(data)
    #upperbands, lowerbands = bollinger_bands(data, 10, 2)
    #st = stochastic_oscillator(data, 10)
 
   
   
    

main()



