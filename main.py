import pickle
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import copy

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
def daily_return(data_frames):
    for i, df in enumerate(data_frames):
        data_frames[i]['Daily_return'] = df.Close - df.Open

#calculate a moving average of closing prices with given data and given days(e.g. The window_size argument specifies the number of days over which the average is taken.)
def moving_average(data_frames, window_size):
      
    for i, df in enumerate(data_frames):
        close_prices = df['Close']
        moving_average = close_prices.rolling(window = window_size).mean()
        data_frames[i][str(window_size) + '-day MA'] = moving_average

#calculate standard_devation
def standard_deviation(data_frames, window_size):

    for i, df in enumerate(data_frames):
        close_prices = df['Close']
        std_dev = close_prices.rolling(window =window_size).std()
        data_frames[i][str(window_size) + '-day S.D'] = std_dev

#calculates upper and lower Bollinger Bands, using moving_averge and two standard deviation
#upperband and lower band are the lines that are plotted at a distance of 'k'
#typically window_size is 20 and k is usually 2
def bollinger_bands(data_frames, window_size = 20, k = 2):

    for i,df in enumerate(data_frames):
        close_prices = df['Close']
        moving_average = close_prices.rolling(window_size).mean()
        std_dev = close_prices.rolling(window_size).std()
        upper_band = moving_average + k * std_dev
        lower_band = moving_average - k * std_dev
        data_frames[i][str(window_size) + '-day BOLU'] = upper_band
        data_frames[i][str(window_size) + '-day BOLD'] = lower_band


#Expoenetial Moving Average that gives more weight to recent prices 
#short term tpyically (12 or 26 days) while Long term (50 day or 200 days) 
def ema(data_frames, window_size = 20):

    for i,df in enumerate(data_frames):
        exponetial = df.Close.ewm(span=window_size, adjust = False).mean()
        data_frames[i][str(window_size) + '-day EMA'] = exponetial


# Calculate weighted_moving_average that assigns higher weight to more recent values and lower weight to older values
# 90 period weighted_moving_average
def WMA(dataframes, window_size = 90):

    for i, df in enumerate(dataframes):
        w = np.arange(1, window_size + 1, 1)
        wma = df.Close.rolling(window_size).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
        dataframes[i][str(window_size) + '-day WMA'] = wma

           
# Calculate Momentum that measures the strength of the price trend in an asset 
# typical momentum day is 14 days or 21 days
def momentum(dataframes, window_size = 14):
    
    for i,df in enumerate(dataframes):
        stock_momentum = df.Close.diff(window_size)
        dataframes[i][str(window_size) + '-day Momentum'] = stock_momentum

#Rate of Change
#Measures the percentage change the current closing price and the closing price n-time days ago and ange
#Standard calculation is 12 days
def ROC(dataframes, window_size = 12):
 
    for i,df in enumerate(dataframes):
        stock_roc = ((df.Close - df.Close.shift(window_size)) / df.Close.shift(window_size) * 100)
        dataframes[i][str(window_size) + '-day ROC'] = stock_roc
  

#Relative Strength Index
#Momentum indicator that compares the magnitude of recent gains to recent losses to determine overbought or oversold conditions in an asset.
# typcially window size is 14 days
def RSI(dataframes, window_size = 14):
    
   for i, df in enumerate(dataframes):
        close_prices = df['Close']
        diff = close_prices.diff()
        up, down = diff.copy(), diff.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        gain = up.rolling(window_size).mean()
        loss = down.abs().rolling(window_size).mean()
        rs = gain / loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        dataframes[i]['RSI_' + str(window_size)] = rsi
  

#Stochastic Oscillator
#momentum indicator that compares a security's close price to its price range over a specified number of days
#typically 14-day day
def stochastic_oscillator(dataframes, window_size = 14):

    for i,df in enumerate(dataframes):
        close = df['Close']
        low = df['Low'].rolling(window_size).min()
        high = df['High'].rolling(window_size).max()
        stochastic = (close - low) / (high - low) * 100
        dataframes[i][str(window_size) + '-day Stoch'] = stochastic
        slowStoch = stochastic.rolling(3).mean()
        dataframes[i]['3_day Stoch'] = slowStoch
  

#Calculate Moving Average Convergence Divergence
def MACD(dataframes, fast_day=12, slow_day=26, signal_day=9):
    for i, df in enumerate(dataframes):
        fast_ma = df.Close.ewm(span = fast_day, adjust = False).mean()
        slow_ma = df.Close.ewm(span = slow_day, adjust = False).mean()
        dataframes[i]['MACD'] = fast_ma - slow_ma
        signal_line = df.MACD.ewm(span=signal_day, adjust = False).mean()
        dataframes[i]['signal Line'] = signal_line

#Williams %R  
#momentum oscillator that measures overbought and oversold levels 
#typically 14 days
def williams_R(dataframes, lookback_day = 14):

    for i, df in enumerate(dataframes):
        highest_high = df['High'].rolling(lookback_day).max()
        lowest_low = df['Low'].rolling(lookback_day).min()
        williams_R = (highest_high - df['Close']) / (highest_high - lowest_low) * -100
        dataframes[i][str(lookback_day) + '-day William %R'] = williams_R

#Aroon indictor
#determine if an stock is in a trend and how strong the trend is
def aroon_indicator(dataframes, window_size = 25):

    for i, df in enumerate(dataframes):
        aroon_high = df['High'].rolling(min_periods = window_size, window = window_size, center = False).apply(np.argmax)
        aroon_low = df['Low'].rolling(min_periods = window_size, window = window_size, center = False).apply(np.argmin)
        aroon_up = (((window_size - (aroon_high + 1)) / window_size)) * 100 
        aroon_down = (((window_size - (aroon_low + 1)) / window_size)) * 100 
        dataframes[i][str(window_size) + '-day 25 Aroon Up'] = aroon_up
        dataframes[i][str(window_size) + '-day 25 Aroon Down'] = aroon_down


#The average directional movement index
#typicall 14 days
#append Positive, Negative direction indicator and Average True range
def ADX(dataframes, window_size = 14):

    for i, df in enumerate(dataframes):
        high = df.High
        low = df.Low
        close = df.Close
        true_range = np.maximum(high - low, np.abs(high - close.shift(1)), np.abs(low - close.shift(1)))
        average_true_range = true_range.rolling(window_size).mean()
        pos_direc_mv = (high - high.shift(1))
        neg_direc_mv = (low - low.shift(1))
        pos_direc_mv[pos_direc_mv < 0] = 0
        neg_direc_mv[neg_direc_mv > 0] = 0
        pos_direc_indi = 100 * (pos_direc_mv.ewm(alpha = 1/window_size).mean() / average_true_range)
        neg_direc_indi = abs(100 * (neg_direc_mv.ewm(alpha = 1/window_size).mean() / average_true_range))
        dx = (abs(pos_direc_indi - neg_direc_indi) / abs(pos_direc_indi + neg_direc_indi)) * 100
        adx = ((dx.shift(1) * (window_size - 1)) + dx) / window_size
        adx_smooth = adx.ewm(alpha = 1/window_size).mean()
        dataframes[i]['ATR'] = average_true_range
        dataframes[i]['Pos_DI'] = pos_direc_indi
        dataframes[i]['Neg_dI'] = neg_direc_indi
        dataframes[i][str(window_size) + '-day ADX'] = adx_smooth

#Commodity Channel Index
#relatively high when prices are far from average
def CCI(data_frames, window_size = 20):
    
    for i, df in enumerate(data_frames):
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        tp_avg = typical_price.rolling(window_size).mean()
        mean_deviation =  abs(typical_price - tp_avg).rolling(window_size).mean()
        cci = (typical_price - tp_avg) / (0.015 * mean_deviation)
        data_frames[i][str(window_size)+ '-day CCI'] = cci


#Coppock curve to determine if it is bull market
#Typciall applied roc14 is 14period(months) = 420 and roc11 is 11 period(months) = 330
#10 months of weighted average
def coppock_curve(data_frames, wma_period = 300, roc14_period = 420, roc11_period = 330):

    def get_wma(data, window_size):
        w = np.arange(1, window_size + 1, 1)
        wma = data.rolling(window_size).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
        return wma
    
    def get_roc(data, window_size):
        stock_roc = ((data - data.shift(window_size)) / data.shift(window_size) * 100)
        return stock_roc

    for i, df in enumerate(data_frames):
        long_roc = get_roc(df.Close, roc14_period)
        short_roc = get_roc(df.Close, roc11_period)
        ROC = long_roc + short_roc
        coppock = get_wma(ROC, wma_period)
        data_frames[i]['Coppock Curve'] = coppock

#On-balance-Volum(OBV)
def on_balance_volume(data_frames):
    
    for i, df in enumerate(data_frames):
        data_frames[i]["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum()

#Chaikin Money Flow(CMF) (MF = money flow)
#typically 20 period
def chaikin_money_flow(data_frames):

    for i, df in enumerate(data_frames):
        MF_Multiplier = ((df.Close - df.Low) - (df.High - df.Close)) / (df.High - df.Low)
        MF_Volume = MF_Multiplier * df.Volume
        data_frames[i]['CMF'] = MF_Volume.rolling(21).mean() / df.Volume.rolling(21).mean()

#Fibonacci retracement levels(FRL)
#The Fibanoacci Ratios are typically 0.236 0.382 0.618 
def Fibonacci_retracement_levels(data_frames):

    for i, df in enumerate(data_frames):
        maximum = df.Close.max()
        minimum = df.Close.min()
        diff = maximum - minimum
        data_frames[i]['FRL_lev1'] = maximum - 0.236 * diff
        data_frames[i]['FRL_lev2'] = maximum - 0.382 * diff
        data_frames[i]['FRL_lev3'] = maximum - 0.618 * diff

#Money Flow Index(MFI)
#MFI above 80 considred overbought and below 20 oversold
def money_flow_index(data_frames, period = 14):


    def get_mfi(df, period = 14):
        typical_price = (df['Close'] + df['High'] + df['Low']) / 3
        money_flow = typical_price * df['Volume']
        positive_flow = [money_flow[i-1] if typical_price[i] > typical_price[i-1] else 0 for i in range(1, len(typical_price))]
        negative_flow = [money_flow[i-1] if typical_price[i] < typical_price[i-1] else 0 for i in range(1, len(typical_price))]
        positive_mf = [sum(positive_flow[i + 1- period : i+1]) for i in range(period-1, len(positive_flow))]
        negative_mf = [sum(negative_flow[i + 1- period : i+1]) for i in range(period-1, len(negative_flow))]
        MFI = 100 * (np.array(positive_mf) / (np.array(positive_mf) + np.array(negative_mf)))
        df.loc[period:, 'MFI'] = MFI

    for i, df in enumerate(data_frames):
        get_mfi(df)





    
#-------------------------------------------------------------------------------Mainfunction-----------------------------------------------------------------------------------
def main():
    path = os.getcwd()
    filename = os.path.join(path, 'testing_set2.pkl')

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    original_data = data
    #copy from original data so that it doesn't modify the original data
    data_frame = copy.deepcopy(original_data)
    
    #Task1
    #closed_percentage = daily_percentage_change_for_close(data)
    #thresholds = find_thresholds(closed_percentage)
    #levels = divide_into_levels(closed_percentage, thresholds)
    #threshold_number_of_data_points(levels, thresholds)

    #Taks2
    
    daily_return(data_frame)
    moving_average(data_frame, 20)
    moving_average(data_frame, 50)
    moving_average(data_frame, 200)
    standard_deviation(data_frame,20)
    WMA(data_frame)
    bollinger_bands(data_frame)
    ema(data_frame)
    momentum(data_frame)
    ROC(data_frame)
    RSI(data_frame)
    MACD(data_frame)
    stochastic_oscillator(data_frame)
    williams_R(data_frame)
    ADX(data_frame)
    CCI(data_frame)
    coppock_curve(data_frame)
    aroon_indicator(data_frame)
    on_balance_volume(data_frame)
    chaikin_money_flow(data_frame)
    Fibonacci_retracement_levels(data_frame)
    money_flow_index(data_frame)
    

    #replace all the NaN to 0
    for i in range(len(data_frame)):
        data_frame[i].fillna(0, inplace = True)
  
    print(data_frame[0].tail(5))
    

    #TO check data is correct
    df = pd.DataFrame(data[0])
    

  
        
    


   
    
       
    
    
  
   

main()



