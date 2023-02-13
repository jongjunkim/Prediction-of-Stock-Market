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
def daily_return(data_frame):
    for i, df in enumerate(data_frame):
        data_frame[i]['Daily_return'] = df.Close - df.Open

#daily log return
def daily_log_return(data_frame):
    
    for i, df in enumerate(data_frame):
        data_frame[i]['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

#calculate a moving average of closing prices with given data and given days(e.g. The window_size argument specifies the number of days over which the average is taken.)
def moving_average(data_frame, window_size):
      
    for i, df in enumerate(data_frame):
        close_prices = df['Close']
        moving_average = close_prices.rolling(window = window_size).mean()
        data_frame[i][str(window_size) + '-day MA'] = moving_average

#calculate standard_devation
def standard_deviation(data_frame, window_size):

    for i, df in enumerate(data_frame):
        close_prices = df['Close']
        std_dev = close_prices.rolling(window =window_size).std()
        data_frame[i][str(window_size) + '-day S.D'] = std_dev

#Donchian_Channel consists of Lower Channel, Upper Channel, and Middle Channel
#typically 20 days
def donchian_channel(data_frame, window_size = 20):

    for i, df in enumerate(data_frame):
        data_frame[i]['Higher_Channel'] = df.High.rolling(window_size).max()
        data_frame[i]['Lower_Channel'] = df.Low.rolling(window_size).min()
        data_frame[i]['Middle_Channel'] = (df.Higher_Channel + df.Lower_Channel) / 2



#calculates upper and lower Bollinger Bands, using moving_averge and two standard deviation
#upperband and lower band are the lines that are plotted at a distance of 'k'
#typically window_size is 20 and k is usually 2
def bollinger_bands(data_frame, window_size, k = 2):

    for i,df in enumerate(data_frame):
        close_prices = df['Close']
        moving_average = close_prices.rolling(window_size).mean()
        std_dev = close_prices.rolling(window_size).std()
        upper_band = moving_average + k * std_dev
        lower_band = moving_average - k * std_dev
        data_frame[i][str(window_size) + '-day BOLU'] = upper_band
        data_frame[i][str(window_size) + '-day BOLD'] = lower_band


#Expoenetial Moving Average that gives more weight to recent prices 
#short term tpyically (12 or 26 days) while Long term (50 day or 200 days) 
def ema(data_frame, window_size):

    for i,df in enumerate(data_frame):
        exponetial = df.Close.ewm(span=window_size, adjust = False).mean()
        data_frame[i][str(window_size) + '-day EMA'] = exponetial


# Calculate weighted_moving_average that assigns higher weight to more recent values and lower weight to older values
# 90 period weighted_moving_average
def WMA(data_frames, window_size = 90):

    for i, df in enumerate(data_frames):
        w = np.arange(1, window_size + 1, 1)
        wma = df.Close.rolling(window_size).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
        data_frames[i][str(window_size) + '-day WMA'] = wma

           
# Calculate Momentum that measures the strength of the price trend in an asset 
# typical momentum day is 14 days or 21 days
def momentum(data_frames, window_size = 14):
    
    for i,df in enumerate(data_frames):
        stock_momentum = df.Close.diff(window_size)
        data_frames[i][str(window_size) + '-day Momentum'] = stock_momentum

#Rate of Change
#Measures the percentage change the current closing price and the closing price n-time days ago and ange
#Standard calculation is 12, 25, 200
def ROC(data_frames, window_size):
 
    for i,df in enumerate(data_frames):
        stock_roc = ((df.Close - df.Close.shift(window_size)) / df.Close.shift(window_size) * 100)
        data_frames[i][str(window_size) + '-day ROC'] = stock_roc
  

#Relative Strength Index
#Momentum indicator that compares the magnitude of recent gains to recent losses to determine overbought or oversold conditions in an asset.
# typcially window size is 14 days
def RSI(data_frames, window_size = 14):
    
   for i, df in enumerate(data_frames):
        close_prices = df['Close']
        diff = close_prices.diff()
        up, down = diff.copy(), diff.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        gain = up.rolling(window_size).mean()
        loss = down.abs().rolling(window_size).mean()
        rs = gain / loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        data_frames[i]['RSI_' + str(window_size)] = rsi
  

#Stochastic Oscillator
#momentum indicator that compares a security's close price to its price range over a specified number of days
#if below 20% oversold, above 80% overbought
#typically 14-day day
def stochastic_oscillator(data_frames, window_size = 14):

    for i,df in enumerate(data_frames):
        close = df['Close']
        low = df['Low'].rolling(window_size).min()
        high = df['High'].rolling(window_size).max()
        stochastic = (close - low) / (high - low) * 100
        data_frames[i][str(window_size) + '-day Stoch'] = stochastic
        slowStoch = stochastic.rolling(3).mean()
        data_frames[i]['3_day Stoch'] = slowStoch

#Awesome Oscillator(AO)
#shows what's happening to the market driving force at the present moment
def awesome_Oscillator(data_frames, short_period = 5, long_period = 34):

    for i, df in enumerate(data_frames):
        median_price = (df.High + df.Low)/2
        data_frames[i]['AO'] = median_price.rolling(short_period).mean() - median_price.rolling(long_period).mean()
  

#Calculate Moving Average Convergence Divergence
def MACD(data_frames, fast_day=12, slow_day=26, signal_day=9):
    for i, df in enumerate(data_frames):
        fast_ma = df.Close.ewm(span = fast_day, adjust = False).mean()
        slow_ma = df.Close.ewm(span = slow_day, adjust = False).mean()
        data_frames[i]['MACD'] = fast_ma - slow_ma
        signal_line = df.MACD.ewm(span=signal_day, adjust = False).mean()
        data_frames[i]['signal Line'] = signal_line

#Williams %R  
#momentum oscillator that measures overbought and oversold levels 
#typically 14 days
def williams_R(data_frames, lookback_day = 14):

    for i, df in enumerate(data_frames):
        highest_high = df['High'].rolling(lookback_day).max()
        lowest_low = df['Low'].rolling(lookback_day).min()
        williams_R = (highest_high - df['Close']) / (highest_high - lowest_low) * -100
        data_frames[i][str(lookback_day) + '-day William %R'] = williams_R

#Aroon indictor
#determine if an stock is in a trend and how strong the trend is
def aroon_indicator(data_frames, window_size = 25):

    for i, df in enumerate(data_frames):
        aroon_high = df['High'].rolling(min_periods = window_size, window = window_size, center = False).apply(np.argmax)
        aroon_low = df['Low'].rolling(min_periods = window_size, window = window_size, center = False).apply(np.argmin)
        aroon_up = (((window_size - (aroon_high + 1)) / window_size)) * 100 
        aroon_down = (((window_size - (aroon_low + 1)) / window_size)) * 100 
        data_frames[i][str(window_size) + '-day 25 Aroon Up'] = aroon_up
        data_frames[i][str(window_size) + '-day `25` Aroon Down'] = aroon_down
        data_frames[i]['Aroon Oscillator'] = aroon_up - aroon_down



#Pivot points (pivot point, Resistance1, Resistance2, Support1, Support2)
def pivot_points(data_frame):

    for i,df in enumerate(data_frame):
        data_frame[i]['pivot_point'] = (df['High'] + df['Low'] + df['Close']) / 3
        data_frame[i]['Resistance1'] = (df.pivot_point * 2) - df.Low
        data_frame[i]['Resistance2'] = (df.pivot_point + (df.High - df.Low))
        data_frame[i]['Support1'] = (df.pivot_point * 2) - df.High
        data_frame[i]['Supprot2'] = (df.pivot_point -  (df.High - df.Low))

#Commodity Channel Index
#relatively high when prices are far from average
def CCI(data_frame, window_size = 20):
    
    for i, df in enumerate(data_frame):
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        tp_avg = typical_price.rolling(window_size).mean()
        mean_deviation =  abs(typical_price - tp_avg).rolling(window_size).mean()
        cci = (typical_price - tp_avg) / (0.015 * mean_deviation)
        data_frame[i][str(window_size)+ '-day CCI'] = cci


#Coppock curve to determine if it is bull market
#Typciall applied roc14 is 14period(months) = 420 and roc11 is 11 period(months) = 330
#10 months of weighted average
def coppock_curve(data_frame, wma_period = 300, roc14_period = 420, roc11_period = 330):

    def get_wma(data, window_size):
        w = np.arange(1, window_size + 1, 1)
        wma = data.rolling(window_size).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
        return wma
    
    def get_roc(data, window_size):
        stock_roc = ((data - data.shift(window_size)) / data.shift(window_size) * 100)
        return stock_roc

    for i, df in enumerate(data_frame):
        long_roc = get_roc(df.Close, roc14_period)
        short_roc = get_roc(df.Close, roc11_period)
        ROC = long_roc + short_roc
        coppock = get_wma(ROC, wma_period)
        data_frame[i]['Coppock Curve'] = coppock

#On balance Volume(OBV) and On_balance_Volume_Mean
def on_balance_volume(data_frame):
    
    for i, df in enumerate(data_frame):
        data_frame[i]['OBV'] = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum()
        data_frame[i]['OBV_Mean'] = df.OBV.rolling(14).mean()




#Chaikin Money Flow(CMF) (MF = money flow)
#typically 20 period
def chaikin_money_flow(data_frame):

    for i, df in enumerate(data_frame):
        data_frame[i]['MF_Multiplier'] = ((df.Close - df.Low) - (df.High - df.Close)) / (df.High - df.Low)
        data_frame[i]['MF_Volume'] = df.MF_Multiplier * df.Volume
        data_frame[i]['CMF'] = df.MF_Volume.rolling(21).mean() / df.Volume.rolling(21).mean()

#accumulation_distribution(ADI)
#use relationship between the stock's price and volume flow
def accumulation_distribution(data_frame):

    for i, df in enumerate(data_frame):
        data_frame[i]['ADI'] = df.MF_Volume.cumsum()


#Fibonacci retracement levels(FRL)
#The Fibanoacci Ratios are typically 0.236 0.382 0.618 
def Fibonacci_retracement_levels(data_frame):

    for i, df in enumerate(data_frame):
        maximum = df.Close.max()
        minimum = df.Close.min()
        diff = maximum - minimum
        data_frame[i]['FRL_lev1'] = maximum - 0.236 * diff
        data_frame[i]['FRL_lev2'] = maximum - 0.382 * diff
        data_frame[i]['FRL_lev3'] = maximum - 0.618 * diff

#Money Flow Index(MFI)
#MFI above 80 considred overbought and below 20 oversold
def money_flow_index(data_frame, period = 14):

    def get_mfi(df, period = 14):
        typical_price = (df['Close'] + df['High'] + df['Low']) / 3
        money_flow = typical_price * df['Volume']
        positive_flow = [money_flow[i-1] if typical_price[i] > typical_price[i-1] else 0 for i in range(1, len(typical_price))]
        negative_flow = [money_flow[i-1] if typical_price[i] < typical_price[i-1] else 0 for i in range(1, len(typical_price))]
        positive_mf = [sum(positive_flow[i + 1- period : i+1]) for i in range(period-1, len(positive_flow))]
        negative_mf = [sum(negative_flow[i + 1- period : i+1]) for i in range(period-1, len(negative_flow))]
        MFI = 100 * (np.array(positive_mf) / (np.array(positive_mf) + np.array(negative_mf)))
        df.loc[period:, 'MFI'] = MFI

    for i, df in enumerate(data_frame):
        get_mfi(df)

#the Triple Exponential Moving Average(TEMA or sometimes called TRIX)
def TEMA(data_frame, window_size = 15):

    for i, df in enumerate(data_frame):
        exponetial1 = df.Close.ewm(span=window_size, adjust = False).mean()
        exponetial2 = exponetial1.ewm(span = window_size, adjust = False).mean()
        exponetial3 = exponetial1.ewm(span = window_size, adjust = False).mean()
        data_frame[i]['TEMA'] = (3*exponetial1) - (3*exponetial2) + exponetial3

#Volume Price Trend indicator, sometimes known as the Price_Volume period
#also called Volume Price Trend
def VPT(data_frame):

    for i, df in enumerate(data_frame):
        close_diff = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)
        vpt = df['Volume'] * close_diff
        vpt = vpt.cumsum()
        data_frame[i]['VPT'] = vpt

#Average Daily Trading Volume
#typcially 20days or 30days
def ADTV(data_frame, window_size = 30):

    for i,df in enumerate(data_frame):
        data_frame[i][str(window_size) + 'day ADTV'] = df.Volume.rolling(window_size).sum()/window_size


#Volume Weighted Average Price
def VWAP(data_frame):

    for i, df in enumerate(data_frame):
        cumulative_tp = df.pivot_point.cumsum()
        cumulative_volume = df.Volume.cumsum()
        data_frame[i]['VWAP'] = cumulative_tp/cumulative_volume

#ICHIMOKU Cloud Trading
#Tenkan_sen = Conversion Line, Kijun_sen = Base line, Senkou span= Leading Span 
#Ichimoku cloud is the area where located between Senkou span1 and span2
def ichimoku_cloud(data_frame):

    for i, df in enumerate(data_frame):
        high_9 = df.High.rolling(window = 9).max()
        low_9 = df.Low.rolling(window = 9).min()
        high_26 = df.High.rolling(window = 26).max()
        low_26 = df.Low.rolling(window = 26).min()
        high_52 = df.High.rolling(window = 52).max()
        low_52 = df.Low.rolling(window = 52).min()
        data_frame[i]['Tenkan_sen'] = (high_9 + low_9)/2
        data_frame[i]['Kijun_sen'] = (high_26 + low_26)/2
        data_frame[i]['Senkou_span1'] = ((df.Tenkan_sen + df.Kijun_sen) / 2).shift(26)
        data_frame[i]['Senkou_span2'] = ((high_52 + low_52) / 2).shift(26)

#Disparity index
#measures the relative position of an aseet's most recent closing price
#value greater than zero- asset is gaining upward momentum, value less than zero- a sign that selling pressure is increasing
def disparity(data_frame, window_size = 14):

    for i, df in enumerate(data_frame):
        mv = df.Close.rolling(window_size).mean()
        data_frame[i]['Disparity'] = (df.Close - mv)/(mv * 100)


#detrended price oscillaotor
def DPO(data_frame, window_size = 20):

    for i, df in enumerate(data_frame):
        detrended_period = int(window_size/2 + 1)
        data_frame[i]['DPO'] = df['Close'].shift(detrended_period)  - df['Close'].rolling(window_size).mean()

#Ease of Movement
def EMV(data_frame, window_size = 14):

    for i,df in enumerate(data_frame):
        distance_moved = (df.High + df.Low)/2 - (df.High.shift(1) + df.Low.shift(1))/2
        box_ratio = (df.Volume/1000000) / (df.High - df.Low)
        emv = distance_moved/box_ratio
        data_frame[i][str(window_size) + 'day EMV'] = emv.rolling(window_size).mean()

#Force Index(FI)
def ForceIndex(data_frame, window_size = 14):

    for i,df in enumerate(data_frame):
        forceindex = (df.Close - df.Close.shift(1)) * df.Volume
        data_frame[i]['1day_FI'] = forceindex
        data_frame[i][str(window_size) + 'day FI'] = forceindex.ewm(span = window_size, adjust = False).mean()

#Keltner channel(KC)
def Keltner_channel(data_frame, window_size = 14):

    for i,df in enumerate(data_frame):
        data_frame[i]['KC_middle'] = df.Close.ewm(span = window_size, adjust = False).mean()
        data_frame[i]['KC_upper'] = df.KC_middle + df.MF_Multiplier * df.ATR
        data_frame[i]['KC_lower'] = df.KC_middle - df.MF_Multiplier * df.ATR

#Know Sure Thing(KST)
def Know_sure_thing(data_frame):

    for i,df in enumerate(data_frame):
        ROC_10 = ((df.Close - df.Close.shift(10)) / df.Close.shift(10) * 100)
        ROC_15 = ((df.Close - df.Close.shift(15)) / df.Close.shift(15) * 100)
        ROC_20 = ((df.Close - df.Close.shift(20)) / df.Close.shift(20) * 100)
        ROC_30 = ((df.Close - df.Close.shift(30)) / df.Close.shift(30) * 100)
        RCMA1 = ROC_10.rolling(10).mean()
        RCMA2 = ROC_15.rolling(10).mean()
        RCMA3 = ROC_20.rolling(10).mean()
        RCMA4 = ROC_30.rolling(15).mean()
        data_frame[i]['KST'] = RCMA1 + RCMA2*2 + RCMA3*3 + RCMA4*4

#mass index
def Mass_index(data_frame, window_size = 9):

    for i,df in enumerate(data_frame):
        difference = df.High - df.Low
        ema9 = difference.ewm(span = window_size, adjust = False).mean()
        ema9_of = ema9.ewm(span= window_size, adjust = False).mean()
        mass = ema9 / ema9_of
        data_frame[i]['MassIndex'] = mass.rolling(25).sum()


#True Strength Index(TSI)
def True_strength_index(data_frame, short_window=13, long_window=25):
    
    for i, df in enumerate(data_frame):
        close_diff = df.Close - df.Close.shift(1)
        pcs = close_diff.ewm(span = long_window, adjust = False).mean()
        pcds = pcs.ewm(span = short_window, adjust = False).mean()
        abs_close_diff = abs(close_diff)
        apcs = abs_close_diff.ewm(span = long_window, adjust = False).mean()
        apcds = apcs.ewm(span = short_window, adjust = False).mean()
        data_frame[i]['TSI'] = pcds/apcds * 100

#The average directional movement index
#typicall 14 days
#append Positive, Negative direction indicator and Average True range
def ADX(data_frames, window_size = 14):

    for i, df in enumerate(data_frames):
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
        data_frames[i]['TR'] = true_range
        data_frames[i]['ATR'] = average_true_range
        data_frames[i]['Pos_DI'] = pos_direc_indi
        data_frames[i]['Neg_dI'] = neg_direc_indi
        data_frames[i][str(window_size) + '-day ADX'] = adx_smooth

#Ultimate Oscillator
def ultimate_oscillator(data_frame):

    for i, df in enumerate(data_frame):
        buying_pressure = df.Close - np.minimum(df.Low, df.Close.shift(1))
        true_range = df.TR
        average7 = buying_pressure.rolling(7).sum() / true_range.rolling(7).sum()
        average14 = buying_pressure.rolling(14).sum() / true_range.rolling(14).sum()
        average28 = buying_pressure.rolling(28).sum() / true_range.rolling(28).sum()
        uo = (((average7*4) + (average14*2) + average28) / (4+2+1)) * 100
        data_frame[i]['UO'] = uo

#Vortext indicator consists of plus and minus (VIN_pos and VIN_neg)
#VM+ and VM- are uptrend and downtrend movement respectively 
def vortex_indicator(data_frame, window_size = 14):

    for i, df in enumerate(data_frame):
        true_range = df.TR
        data_frame[i]['VM_plus'] = abs(df.High - df.Low.shift(1))
        data_frame[i]['VM_minus'] = abs(df.Low - df.High.shift(1))
        sum_tr = true_range.rolling(window_size).sum()
        sum_vm_plus = df.VM_plus.rolling(window_size).sum()
        sum_vm_minus = df.VM_minus.rolling(window_size).sum()
        data_frame[i]['VIN_pos'] = sum_vm_plus/sum_tr
        data_frame[i]['VIN_neg'] = sum_vm_minus/sum_tr



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

    #Indicators
    daily_return(data_frame)
    daily_log_return(data_frame)
    standard_deviation(data_frame,20)
    ROC(data_frame, 12)
    ROC(data_frame, 25)
    ROC(data_frame, 200)
    RSI(data_frame)

    #Moving average 20, 50 ,and 200 periods
    moving_average(data_frame, 20)
    moving_average(data_frame, 50)
    moving_average(data_frame, 200)

    #short term Exponential Moving average(EMA)
    ema(data_frame, 12)
    ema(data_frame, 26)
    #long term EMA
    ema(data_frame, 50)
    ema(data_frame, 200)

    #Technical Indicators
    WMA(data_frame)
    bollinger_bands(data_frame, 20)
    bollinger_bands(data_frame, 50)
    momentum(data_frame)
    True_strength_index(data_frame)
    ADX(data_frame)
    MACD(data_frame)
    stochastic_oscillator(data_frame)
    awesome_Oscillator(data_frame)
    williams_R(data_frame)
    CCI(data_frame)
    coppock_curve(data_frame)
    aroon_indicator(data_frame)
    on_balance_volume(data_frame)
    ADTV(data_frame)
    Fibonacci_retracement_levels(data_frame)
    money_flow_index(data_frame)
    TEMA(data_frame)
    VPT(data_frame)
    pivot_points(data_frame)
    VWAP(data_frame)
    ichimoku_cloud(data_frame)
    disparity(data_frame)
    DPO(data_frame)
    donchian_channel(data_frame)
    EMV(data_frame)
    vortex_indicator(data_frame)
    ultimate_oscillator(data_frame)
    chaikin_money_flow(data_frame)
    accumulation_distribution(data_frame)
    ForceIndex(data_frame)
    Keltner_channel(data_frame)
    Know_sure_thing(data_frame)
    Mass_index(data_frame)
    
    
   
    
    #replace all the NaN to 0
    for i in range(len(data_frame)):
        data_frame[i].fillna(0, inplace = True)
  
    print(data_frame[0].tail(5))
 

    

    #TO check data is correct
    df = pd.DataFrame(data[0])

    
    
    


   
    
       
    
    
  
   

main()



