import numpy as np
import pandas as pd

# Load in the environment variables for where we store the datasets and how much history to source
from dotenv import load_dotenv
import os
load_dotenv()
data_dir = os.getenv('DATA_DIR')
historical_data = os.getenv('HISTORICAL_DATA')

# Turn off the intrusive performance warnings
# Approach taken from stackoverflow.com with the setting of copy_on_write to be true suggested by pandas itself within the warning
# https://stackoverflow.com/questions/51521526/python-pandas-how-to-supress-performancewarning
pd.options.mode.copy_on_write = True
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

def source_market_data(api_object, stocks):
    '''
    A function to source the market data from the Yahoo Finance API
    :param api_object: the Yahoo Finance API object
    :param stocks: the list of stock tickers to source
    :return: none
    '''
    print('Sourcing the EOD market data')
    raw = api_object.history(period=historical_data)
    # We just need the close data
    market_data = raw['Close']
    # Set the date to be the index
    market_data.index = market_data.index.date
    market_data.index.names = ['date']
    # Calculate the day over day return
    for ticker in stocks:
        market_data[ticker + '_Return'] = np.log(market_data[ticker] / market_data[ticker].shift(1))
    market_data.dropna(inplace=True)
    # Save the data down
    market_data.to_csv(data_dir + 'market_data.csv')

def source_metrics(ticker):
    '''
    A function to source the technical metrics from the Alpha Vantage API.
    Approach taken as outlined in the API documentation at https://www.alphavantage.co/documentation/.
    The code loads the data, sets the index and adds the ticker to the start of each column
    :param ticker: the list of stock tickers to source
    :return: individual pandas dataframe objects containing the sourced information for each technical metric
    '''
    print('Sourcing technical metrics for ' + ticker)
    # Source AD
    url = 'https://www.alphavantage.co/query?function=AD&symbol=' + ticker + '&interval=daily&datatype=csv&apikey=I2UR68ST032EG0J5'
    AD = pd.read_csv(url)
    AD.set_index('time', inplace=True)
    AD.rename(columns={'Chaikin A/D': ticker + '_AD'}, inplace=True)
    # Source Ultimate oscillator
    url = 'https://www.alphavantage.co/query?function=ULTOSC&symbol=' + ticker + '&interval=daily&datatype=csv&apikey=I2UR68ST032EG0J5'
    Ult = pd.read_csv(url)
    Ult.set_index('time', inplace=True)
    Ult.rename(columns={'ULTOSC': ticker + '_ULT'}, inplace=True)
    # Source RSI
    url = 'https://www.alphavantage.co/query?function=RSI&symbol=' + ticker + '&interval=daily&time_period=10&series_type=close&datatype=csv&apikey=I2UR68ST032EG0J5'
    RSI = pd.read_csv(url)
    RSI.set_index('time', inplace=True)
    RSI.rename(columns={'RSI': ticker + '_RSI'}, inplace=True)
    # Source Williams
    url = 'https://www.alphavantage.co/query?function=WILLR&symbol=' + ticker + '&interval=daily&time_period=10&datatype=csv&apikey=I2UR68ST032EG0J5'
    WIL = pd.read_csv(url)
    WIL.set_index('time', inplace=True)
    WIL.rename(columns={'WILLR': ticker + '_WIL'}, inplace=True)
    # Source Average Directional Movement Index (ADX)
    url = 'https://www.alphavantage.co/query?function=ADX&symbol=' + ticker + '&interval=daily&time_period=10&datatype=csv&apikey=I2UR68ST032EG0J5'
    ADX = pd.read_csv(url)
    ADX.set_index('time', inplace=True)
    ADX.rename(columns={'ADX': ticker + '_ADX'}, inplace=True)
    # Source AROON
    url = 'https://www.alphavantage.co/query?function=AROON&symbol=' + ticker + '&interval=daily&time_period=10&datatype=csv&apikey=I2UR68ST032EG0J5'
    AROON = pd.read_csv(url)
    AROON.set_index('time', inplace=True)
    AROON.rename(columns={'Aroon Down': ticker + '_AR_DN', 'Aroon Up': ticker + '_AR_UP'}, inplace=True)
    # Source Money Flow Index (MFI)
    url = 'https://www.alphavantage.co/query?function=MFI&symbol=' + ticker + '&interval=daily&time_period=10&datatype=csv&apikey=I2UR68ST032EG0J5'
    MFI = pd.read_csv(url)
    MFI.set_index('time', inplace=True)
    MFI.rename(columns={'MFI': ticker + '_MFI'}, inplace=True)
    # Source DX
    url = 'https://www.alphavantage.co/query?function=DX&symbol=' + ticker + '&interval=daily&time_period=10&datatype=csv&apikey=I2UR68ST032EG0J5'
    DX = pd.read_csv(url)
    DX.set_index('time', inplace=True)
    DX.rename(columns={'DX': ticker + '_DX'}, inplace=True)
    # Source SAR
    url = 'https://www.alphavantage.co/query?function=SAR&symbol=' + ticker + '&interval=daily&datatype=csv&apikey=I2UR68ST032EG0J5'
    SAR = pd.read_csv(url)
    SAR.set_index('time', inplace=True)
    SAR.rename(columns={'SAR': ticker + '_SAR'}, inplace=True)
    # Source On Balance Volume (OBV)
    url = 'https://www.alphavantage.co/query?function=OBV&symbol=' + ticker + '&interval=daily&datatype=csv&apikey=I2UR68ST032EG0J5'
    OBV = pd.read_csv(url)
    OBV.set_index('time', inplace=True)
    OBV.rename(columns={'OBV': ticker + '_OBV'}, inplace=True)
    return AD, Ult, RSI, WIL, ADX, AROON, MFI, DX, SAR, OBV

def technical_metrics_batch(stocks):
    '''
    A function to build up the technical metric data table one stock at a time.
    Calls the sourcing function for each one then concatenates into the overall dataframe
    :param stocks: the list of stock tickers to source
    :return: none
    '''
    df = pd.DataFrame()
    for name in stocks:
        # Source technical metrics for each stock
        AD, Ult, RSI, WIL, ADX, AROON, MFI, DX, SAR, OBV = source_metrics(name)
        df = pd.concat([df, AD, Ult, RSI, WIL, ADX, AROON, MFI, DX, SAR, OBV], axis=1)
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    # Save down the file
    df.to_csv(data_dir + 'technical_metrics.csv')

def process_data(tickers):
    '''
    A function to create the simple moving averages and trends. Called once we have the market data and technical metrics.
    The approach to digitize feature data by the first and second moments of the distribution comes from the book "Python for Finance"
    (Yves Hilpisch, 2019, Section 15 on Trading Strategies, Page 508)
    :param tickers: the list of stock tickers
    :return: none
    '''
    print('Pulling together the datasets')
    df_tec = pd.read_csv(data_dir + 'technical_metrics.csv', index_col='time')
    df_tec.index.rename('date', inplace=True)
    df_mkt = pd.read_csv(data_dir + 'market_data.csv', index_col='date')
    df_com = pd.merge(df_mkt, df_tec, how='inner', on='date')
    df_com.dropna(inplace=True)

    # Create a short horizon moving average metric on the S&P
    # Allocate to bins based on the mean and standard deviation
    df_com['SNP_SMA_sht'] = df_com['^GSPC'].rolling(window=40).mean()
    mu = df_com['SNP_SMA_sht'].mean()
    v = df_com['SNP_SMA_sht'].std()
    bins = [mu - v, mu, mu + v]
    df_com['SNP_SMA_sht_bin'] = np.digitize(df_com['SNP_SMA_sht'], bins=bins)

    # Create a long horizon moving average metric on the S&P
    # Allocate to bins based on the mean and standard deviation
    df_com['SNP_SMA_lng'] = df_com['^GSPC'].rolling(window=160).mean()
    mu = df_com['SNP_SMA_lng'].mean()
    v = df_com['SNP_SMA_lng'].std()
    bins = [mu - v, mu, mu + v]
    df_com['SNP_SMA_lng_bin'] = np.digitize(df_com['SNP_SMA_lng'], bins=bins)

    # Create a short horizon moving average metric on the VIX
    # Allocate to bins based on the mean and standard deviation
    df_com['VIX_SMA_sht'] = df_com['^VIX'].rolling(window=40).mean()
    mu = df_com['VIX_SMA_sht'].mean()
    v = df_com['VIX_SMA_sht'].std()
    bins = [mu - v, mu, mu + v]
    df_com['VIX_SMA_sht_bin'] = np.digitize(df_com['VIX_SMA_sht'], bins=bins)

    # Create a long horizon moving average metric on the VIX
    # Allocate to bins based on the mean and standard deviation
    df_com['VIX_SMA_lng'] = df_com['^VIX'].rolling(window=160).mean()
    mu = df_com['VIX_SMA_lng'].mean()
    v = df_com['VIX_SMA_lng'].std()
    bins = [mu - v, mu, mu + v]
    df_com['VIX_SMA_lng_bin'] = np.digitize(df_com['VIX_SMA_lng'], bins=bins)

    for ticker in tickers:
        print('Processing the technical metrics for ' + ticker)
        # Process the AD : Look at the trend
        # If the Accumulation/Distribution Line is trending upward it indicates that the price may follow
        source_col = '{}_AD'.format(ticker)
        new_col = '{}_AD_trend'.format(ticker)
        df_com[new_col] = np.sign(df_com[source_col].diff(periods=3))
        # Process Ultimate oscillator : Ranges from 0 to 100
        # Interpreted as an overbought/oversold indicator when the value is over 70/below 30
        bins = [0, 40, 60, 100]
        source_col = '{}_ULT'.format(ticker)
        new_col = '{}_ULT_BIN'.format(ticker)
        df_com[new_col] = np.digitize(df_com[source_col], bins=bins)
        # Process RSI : Ranges from 0 to 100
        # The RSI is interpreted as an overbought/oversold indicator when the value is over 70/below 30
        bins = [0, 40, 60, 100]
        source_col = '{}_RSI'.format(ticker)
        new_col = '{}_RSI_BIN'.format(ticker)
        df_com[new_col] = np.digitize(df_com[source_col], bins=bins)
        # Process Williams : Ranges from 0 to 100
        # Inverted scale. Sell signal when crosses 20. Buy signal when crosses 80
        bins = [0, -20, -80, -100]
        source_col = '{}_WIL'.format(ticker)
        new_col = '{}_WIL_BIN'.format(ticker)
        df_com[new_col] = np.digitize(df_com[source_col], bins=bins)
        # Process Average Directional Movement Index (ADX) : Ranges from 0 to 100 but rarely above 60
        # High number is a strong trend. Low number is a weak trend
        bins = [0, 20, 40, 60, 100]
        source_col = '{}_ADX'.format(ticker)
        new_col = '{}_ADX_BIN'.format(ticker)
        df_com[new_col] = np.digitize(df_com[source_col], bins=bins)
        # Process AROON : Up and Down indicators
        # Up between 70 and 100 indicates an upward trend. Down between 70 ans 100 indicates a downward trend
        dn_col = '{}_AR_DN'.format(ticker)
        up_col = '{}_AR_UP'.format(ticker)
        diff_col = '{}_AR_DIFF'.format(ticker)
        df_com[diff_col] = df_com[up_col] - df_com[dn_col]
        bins = [-100, -40, 40, 100]
        final_col = '{}_AR_BIN'.format(ticker)
        df_com[final_col] = np.digitize(df_com[diff_col], bins=bins)
        # Process Money Flow Index (MFI). Ranges from 0 to 100
        # Values above 80 and below 20 indicate market top / bottoms
        bins = [0, 30, 70, 100]
        source_col = '{}_MFI'.format(ticker)
        new_col = '{}_MFI_BIN'.format(ticker)
        df_com[new_col] = np.digitize(df_com[source_col], bins=bins)
        # Process DX. Ranges from 0 to 100 but rarely above 60
        # High number is a strong trend. Low number is a weak trend
        bins = [0, 20, 40, 60, 100]
        source_col = '{}_DX'.format(ticker)
        new_col = '{}_DX_BIN'.format(ticker)
        df_com[new_col] = np.digitize(df_com[source_col], bins=bins)
        # Process On Balance Volume (OBV)
        # Look at the trend
        source_col = '{}_OBV'.format(ticker)
        new_col = '{}_OBV_trend'.format(ticker)
        df_com[new_col] = np.sign(df_com[source_col].diff(periods=3))
        # Calculate the Stop And Reverse flag
        df_com[ticker + '_SAR_Flag'] = np.sign(df_com[ticker] - df_com[ticker + '_SAR'])

        # Create a column holding the direction of the return - will be +1 for positive and -1 for negative
        df_com[ticker + '_Direction'] = np.sign(df_com[ticker + '_Return']).astype(int)
        # Create an additional column remapping this direction indicator to 1 or 0. Used in the xAI functions
        df_com[ticker + '_Direction_bin'] = df_com[ticker + '_Direction'].apply(lambda x: 1 if x > 0 else 0)

        # The approach to create feature data based on lagged returns and digitize by the first and second moments of the distribution comes from the book "Python for Finance" (Yves Hilpisch, 2019)
        # The code below create features based on lagged returns for the last five days and is taken from Section 15 (Trading Strategies) of the book
        lags = 5
        cols = []
        # Create the lagged log returns
        for lag in range(1, lags + 1):
            col = ticker + '_lag_{}'.format(lag)
            df_com[col] = df_com[ticker + '_Return'].shift(lag)
            cols.append(col)
        # Allocate to bins based on the mean and standard deviation
        mu = df_com[ticker + '_Return'].mean()
        v = df_com[ticker + '_Return'].std()
        bins = [mu - v, mu, mu + v]
        for col in cols:
            col_bin = col + '_bin'
            df_com[col_bin] = np.digitize(df_com[col], bins=bins)

        # Create a short horizon moving average metric on the EOD close
        df_com[ticker + '_SMA_sht'] = df_com[ticker].rolling(window=40).mean()
        # Create a long horizon moving average metric on the EOD close
        df_com[ticker + '_SMA_lng'] = df_com[ticker].rolling(window=160).mean()
        # Allocate to bins based on the mean and standard deviation
        mu = df_com[ticker].mean()
        v = df_com[ticker].std()
        bins = [mu - v, mu, mu + v]
        df_com[ticker + '_SMA_sht_bin'] = np.digitize(df_com[ticker + '_SMA_sht'], bins=bins)
        df_com[ticker + '_SMA_lng_bin'] = np.digitize(df_com[ticker + '_SMA_lng'], bins=bins)
    df_com.dropna(inplace=True)
    df_com.to_csv(data_dir + 'final_processed_dataset.csv')

def main():
    '''
    The main function running the data sourcing process
    :return: none
    '''
    import yfinance as yf
    import time
    import json

    # Execution is timed to track how well it scales as the volume of stocks increases
    start_time = time.time()
    # Load in the list of stock tickers from the model_portfolio file
    model_portfolio = pd.read_csv(data_dir + 'model_portfolio.csv')
    core_stocks = model_portfolio["Stock_Ticker"].tolist()
    # Add on the index tickers stored in the environment config file
    index_tickers = json.loads(os.environ['INDEX_TICKERS'])
    all_tickers = core_stocks + index_tickers
    # Create the Yahoo Finance API object
    api_object = yf.Tickers(all_tickers)
    # Source market data
    source_market_data(api_object, all_tickers)
    # Source technical metrics
    technical_metrics_batch(core_stocks)
    # Process the data
    process_data(core_stocks)
    end_time = time.time()
    print('Process execution took %s seconds' % (end_time - start_time))
if __name__ == "__main__":
    main()