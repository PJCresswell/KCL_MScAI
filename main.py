import numpy as np
import pandas as pd

# Import the reporting functions
import reporting

# Load in the environment variables for where we find the data and store the reports and models
from dotenv import load_dotenv
import os
load_dotenv()
data_dir = os.getenv('DATA_DIR')
report_dir = os.getenv('REPORT_DIR')
model_dir = os.getenv('MODEL_DIR')

import json
import pickle
import time

# Turn off the intrusive performance warnings
# Approach taken from stackoverflow.com with the setting of copy_on_write to be true returned by pandas itself within the warning
# https://stackoverflow.com/questions/51521526/python-pandas-how-to-supress-performancewarning
from warnings import simplefilter
simplefilter(action="ignore", category=Warning)

def split_data(all_data):
    '''
    A function to split the model data into training and testing datasets. Configured to be 80% train, 20% test.
    Uses a sequential split (as opposed to a random split) in order to then calculate the daily returns across each window
    :param all_data: the full set of market data sourced
    :return: the individual training and testing datasets
    '''
    split_point = int(len(all_data) * 0.8)
    training_set = all_data.iloc[:split_point].copy()
    testing_set = all_data.iloc[split_point:].copy()
    return training_set, testing_set

def train_model(classifier, train, ticker):
    '''
    A function to train the model.
    Called with different classifiers - used for the production MLP model training as well as the alternative models
    :param classifier: the details of the classifier model to be trained
    :param train: the training dataset
    :param ticker: the ticker of interest
    :return: the model object and the columns used to train the model
    '''
    print('Training ' + str(classifier) + ' model for ' + ticker)
    # Define the columns that we will use to train the model
    columns = ([
        # The bucketed lagged returns
        ticker+'_lag_1_bin', ticker+'_lag_2_bin', ticker+'_lag_3_bin', ticker+'_lag_4_bin', ticker+'_lag_5_bin',
        # The stock based technical metrics
        ticker+'_AD_trend', ticker+'_ULT_BIN', ticker+'_RSI_BIN', ticker+'_WIL_BIN', ticker+'_ADX_BIN',
        ticker+'_AR_BIN', ticker+'_MFI_BIN', ticker+'_DX_BIN', ticker+'_OBV_trend', ticker+'_SAR_Flag',
        # The short and long horizon moving averages
        ticker+'_SMA_sht_bin', ticker+'_SMA_lng_bin', 'SNP_SMA_sht_bin', 'SNP_SMA_lng_bin', 'VIX_SMA_sht_bin', 'VIX_SMA_lng_bin',
    ])
    # Train the model to predict the movement direction using the training dataset
    model = classifier
    model.fit(train[columns], train[ticker + '_Direction_bin'])
    return model, columns

def run_prediction(model, columns, train, test, ticker):
    '''
    A function to run the model to predict the movement direction for that stock. Run across both the training and testing datasets.
    Then calculates the return based on that prediction. Multiplies yesterday's position prediction by today's return.
    Needed to reflect that you take action based on the prediction and then see the results of this action the next day
    :param model: the model object
    :param columns: the columns used to train the model
    :param train: the training dataset
    :param test: the testing dataset
    :param ticker: the ticker of interest
    :return: none
    '''
    print('Running prediction for ' + ticker)

    # Calculate the predicted model position on the testing dataset - will be 0 or 1
    test[ticker + '_model_position'] = model.predict(test[columns])
    # Remap to +1 and -1. Needed by the following step
    # eg if you go short and the return is negative, then the return is positive
    test[ticker + '_model_position'] = test[ticker + '_model_position'].apply(lambda x: 1 if x > 0 else -1)
    # Calculate what this then gives in terms of return on the testing dataset
    test[ticker + '_model_results'] = test[ticker + '_model_position'].shift(1) * test[ticker + '_Return']
    # Calculate the predicted model position on the training dataset - will be 0 or 1
    train[ticker + '_model_position'] = model.predict(train[columns])
    # Remap to +1 and -1. Needed by the following step
    train[ticker + '_model_position'] = train[ticker + '_model_position'].apply(lambda x: 1 if x > 0 else -1)
    # Calculate what this then gives in terms of return on the training dataset
    train[ticker + '_model_results'] = train[ticker + '_model_position'].shift(1) * train[ticker + '_Return']

def explain_ANCHORS(date, stock, model, test, columns, explain_target, forecast):
    '''
    A function to run the Anchor decision rule explain
    :param date: the prediction date
    :param stock: the stock of interest
    :param model: the model object
    :param test: the testing dataset
    :param columns: the columns used to train the model
    :param explain_target: the input data on the prediction date to be explained
    :param forecast: the model movement prediction
    :return: none
    '''
    from alibi.explainers import AnchorTabular

    # Link to the example integration of the ANCHOR functionality using scikit-learn and tabular data which I adapted for my implementation
    # https://docs.seldon.io/projects/alibi/en/stable/examples/anchor_tabular_iris.html

    # There are different explainer options. We are using the tabular explainer as using tabular data
    # Takes a callable prediction function. Implemented for my model using a lamdba function
    predict_fn = lambda x: model.predict(x)
    exp = AnchorTabular(predictor=predict_fn, feature_names=columns)
    dataset = test[columns].to_numpy()
    # Fit the explainer to the test dataset
    # Bins numerical features as given. I have left values to be as per the default
    exp.fit(dataset, disc_perc=(25, 50, 75))
    # Calculate the anchor result for just the explain target to a threshold of 99%
    target = explain_target[columns].to_numpy()
    explanation = exp.explain(target, threshold=0.99)
    anchor_string = '%s' % (' AND '.join(explanation.anchor))
    # Call the function to generate the report
    reporting.anchors_report(date, stock, anchor_string, explanation.precision, explanation.coverage, forecast)

def explain_SHAP(ticker, explain_date, model, train_data, test_data, columns):
    '''
    A function to generate the SHAP feature importance explain
    :param ticker: the ticker of interest
    :param explain_date: the report execution date
    :param model: the model object used for the prediction
    :param train_data: the training dataset
    :param test_data: the testing dataset
    :param columns: the columns used to train the model
    :return: none
    '''
    import shap

    # Link to the example integration of the SHAP functionality using scikit-learn which I adapted for my implementation
    # https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Iris%20classification%20with%20scikit-learn.html
    # My first step is to cut back our 10 years of historical data into 100 samples from the training set
    small_train = shap.sample(train_data, 100)
    # I then create the explainer
    # There are different explainer options. I am using the Kernel explainer
    explainer = shap.KernelExplainer(model.predict, small_train[columns])
    # I now generate the SHAP values for the last 100 days of the test dataset
    small_test = test_data.tail(100)
    # The last row in the resulting SHAP values table will be our explain date
    shap_values = explainer(small_test[columns])
    # I now call the report generation function
    reporting.SHAP_report(shap_values, ticker, explain_date)

def explain_DICE(date, ticker, model, train, columns, explain_target, forecast):
    '''
    A function to run the diverse counterfactuals (DICE) explain
    :param date: the prediction date
    :param ticker: the stock of interest
    :param model: the model object
    :param train: the training dataset
    :param columns: the columns used to train the model
    :param explain_target: the input data on the prediction date to be explained
    :param forecast: the model movement prediction
    :return: none
    '''
    import dice_ml

    # Link to the example integration of the DICE functionality with scikit-learn which I adapted for my implementation
    # https://interpret.ml/DiCE/readme.html#getting-started-with-dice

    # Define the data to be used. Takes a copy of the training data and adds a column with the predicted direction
    # Tells the explainer that this prediction direction is the outcome to be varied
    # Needs to be either 0 or 1 in order to use the "give me the opposite" functionality
    new_train = train[columns + [ticker + '_Direction_bin']].copy()
    d = dice_ml.Data(dataframe=new_train, continuous_features=columns, outcome_name=ticker + '_Direction_bin')
    # Define the model to be used to generate the counterfactuals
    m = dice_ml.Model(model=model, backend="sklearn", model_type="classifier")
    # Create the explainer using the data and model specified
    exp = dice_ml.Dice(d, m, method="random")
    # Store the original values of the input data used by the model
    org = explain_target[columns].values.tolist()

    # If no counterfactuals are possible, an error is raised
    # This is actually a valid outcome and so this error is neatly trapped and reported
    try:
        # First try and generate counterfactuals using all features
        # Generates 5 examples
        e2 = exp.generate_counterfactuals(explain_target[columns], total_CFs=5, desired_class='opposite')
        # Save down
        e2.cf_examples_list[0].final_cfs_df.to_csv(path_or_buf=report_dir + 'counterfactuals_all_' + ticker + '.csv', index=False)
        # Call the function to generate the report with the results
        # Pass in the context that uses all features
        reporting.DICE_report(e2, org, forecast, date, ticker, columns, 'ALL')
    except Exception as ex:
        print("Not possible to generate counterfactuals for " + ticker + " using all features")
    try:
        # Load in the user defined columns which can be varied from the environment configuration
        cf_columns = json.loads(os.environ['CF_COLUMNS'])
        features = [ticker + column for column in cf_columns]
        # Try and generate counterfactuals using the restricted feature list provided by the user
        e3 = exp.generate_counterfactuals(explain_target[columns], total_CFs=5, desired_class='opposite', features_to_vary=features)
        # Save down
        e3.cf_examples_list[0].final_cfs_df.to_csv(path_or_buf=report_dir + 'counterfactuals_filtered_' + ticker + '.csv', index=False)
        # Call the function to generate the report with the results
        # Pass in the context that uses specified features
        reporting.DICE_report(e3, org, forecast, date, ticker, columns, str(features))
    except Exception as ex:
        print("Not possible to generate counterfactuals for " + ticker + " using the restricted list of features")

def explain_Similarity(ticker, train, test, explain_date_data, columns, forecast):
    '''
    A function to find the most similar date in the training dataset to that being explained.
    Views the input data on each date as a vector. Uses cosine similarity to find the closest match.
    Calls the reporting function to generate the report showing details on both the explain and matched dates
    :param ticker: the stock of interest
    :param train: the training dataset
    :param test: the testing dataset
    :param explain_date_data: the input data on the prediction date to be explained
    :param columns: the columns used by the model
    :param forecast: the model movement prediction
    :return: none
    '''
    from sklearn.metrics.pairwise import cosine_similarity

    print('Running similarity explain')
    dir_col = ticker + '_Direction'
    # Filter out from the training dataset just those dates with the same forecast as the explain date
    samples = train[train[dir_col] == forecast]
    # For all dates left, calculate the cosine similarity between this date and the explain date
    result = cosine_similarity(samples[columns], explain_date_data[columns])
    # Find the date with the maximum similarity across all in the training set
    index_max = np.argmax(result)
    # Find the confidence in the match
    confidence = max(result)
    compdate = train[index_max:index_max + 1].index[0]
    explain_date = explain_date_data.index[0]
    print('Explain date : ' + str(explain_date))
    print('Most similar date in training set : ' + str(compdate))
    # Save down
    pd.concat([explain_date_data[columns], train[columns][index_max:index_max + 1]]).to_csv(path_or_buf=report_dir + 'similarity' + ticker + '.csv', index=False)
    # The Similarity report needs various details on the preceding and following dates
    # Defined here
    explain_date_window = test[ticker + '_Return'].tail(7)
    comp_date_previous = train.iloc[index_max - 7 + 1:index_max + 1]
    comp_date_following = train.iloc[index_max:index_max + 7]
    comp_date_values = train.loc[[compdate]]
    comp_date_min1_values = train.shift(1)[train.index == compdate]
    # Finally we call the report
    reporting.daily_report(compdate, ticker, comp_date_values, comp_date_min1_values, 'SIMILAR', explain_date, confidence, explain_date_window, comp_date_previous, comp_date_following, forecast)

def run_challenger_model(train, test, portfolio, date):
    '''
    A function to run the Decision Tree Classifier Challenger Model. Also includes the reporting component
    :param train: the training dataset
    :param test: the testing dataset
    :param portfolio: the portfolio of stocks to be assessed
    :param date: the execution date
    :return: none
    '''
    from sklearn.tree import DecisionTreeClassifier

    #  A Decision Tree Classifier challenger model is trained and a performance report generated for each stock in the portfolio
    for stock in portfolio:
        # Define the model with a maximum depth of 4
        # Needs to be easily explainable so keeps simple
        model = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
        # Train the model
        model_trained, columns_trained = train_model(model, train, stock)
        # Take a copy of the training and testing datasets for processing separate from the main model datasets
        train_cp = train.copy(deep=True)
        test_cp = test.copy(deep=True)
        # Generate the model predictions
        run_prediction(model_trained, columns_trained, train_cp, test_cp, stock)
        # Now generate the report
        reporting.challenger_report(model_trained, columns_trained, train_cp, test_cp, stock, date)

def portfolio_opt(test, core_stocks, date_t, date_tmin1):
    '''
    A function to run the portfolio optimisation analysis
    :param test: the testing dataset
    :param core_stocks: the list of stocks within the portfolio
    :param date_t: the date of execution
    :param date_tmin1: the date before the execution date
    :return: none
    '''
    import yfinance as yf
    from pypfopt import EfficientFrontier
    from pypfopt import risk_models
    from pypfopt import expected_returns
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

    position_results = []
    # Pull out the model generated movement predictions for the last two days
    for stock in core_stocks:
        new_position = test[stock + '_model_position'].tail(1).iloc[0]
        old_position = test[stock + '_model_position'].tail(2).iloc[0]
        position_results.append([stock, date_t, new_position, date_tmin1, old_position])
    # Source the list of stocks and the covariance horizon from the environment variables
    portfolio_stocks = json.loads(os.environ['OPTIMISATION_PORTFOLIO'])
    historical_data = os.getenv('COVARIANCE_HORIZON')
    position_results_df = pd.DataFrame(position_results)

    # The approach and code here comes directly from the PyPortfolioOpt documentation
    # https://pypi.org/project/pyportfolioopt/
    # First, source the historical close data for the stocks in the portfolio
    tickers = yf.Tickers(portfolio_stocks)
    raw = tickers.history(period=historical_data)
    market_data = raw['Close']

    # Next we calculate the expected returns and sample covariance using this data
    mu = expected_returns.mean_historical_return(market_data)
    S = risk_models.sample_cov(market_data)

    # Find the efficient frontier - the set of optimal portfolios which minimise the risk for a target return
    # Allows both long and short positions
    ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
    # Find the portfolio from this set which gives the maximum return for the lowest risk - the maximal Sharpe ratio
    ef.max_sharpe()
    # Helper method provided by the package to set any weights whose absolute values are below the cutoff to zero, and round the rest
    cleaned_weights = ef.clean_weights()
    # Save down the weights
    ef.save_weights_to_file(report_dir + "weights.csv")  # saves to file

    # Use these weights to generate a discrete allocation
    # Firstly, load in the money available from the environment configuration file
    portfolio_value = os.getenv('PORTFOLIO_VALUE')
    latest_prices = get_latest_prices(market_data)
    # Generate the discrete allocation
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=int(portfolio_value))
    allocation, leftover = da.greedy_portfolio()
    # Finally, generate the report
    reporting.portfolio_report(portfolio_stocks, date_t, position_results_df, cleaned_weights, mu, S, ef, portfolio_value, allocation, leftover)

def main():
    '''
    The main batch function. Controls the execution of each individual process using the environment variables
    :return: none
    '''
    from sklearn.neural_network import MLPClassifier

    # Load in the list of stock tickers from the model_portfolio file
    model_portfolio = pd.read_csv(data_dir + 'model_portfolio.csv')
    core_stocks = model_portfolio["Stock_Ticker"].tolist()
    model_portfolio.index = model_portfolio["Stock_Ticker"]

    # Source the model input data
    final_data = pd.read_csv(data_dir + 'final_processed_dataset.csv', index_col='date')

    # Set the prediction date = date_t = the last row in the dataset
    # Also sets the previous date = date t-1, which is used in the reporting
    date_t = final_data.tail(1).index[0]
    date_t_data = final_data.loc[[date_t]]
    date_tmin1 = final_data.tail(2).index[0]
    date_tmin1_data = final_data.loc[[date_tmin1]]

    # Split the input data into testing and training sets
    train, test = split_data(final_data)

    # All of the core processes are controlled in the run by environment variables
    # If set to be True then the process is run

    if os.getenv('TRAINING') == 'True':
        # The process is timed in order to assess how well it
        # scales with volumes
        start_time = time.time()
        # Get the hidden layer settings from the model_portfolio dataframe
        layers = model_portfolio["Hidden_Layers"]
        for stock in core_stocks:
            # Set the hidden layer settings for this specific stock
            architecture = layers.get(stock)
            hidden_layers = json.loads(architecture)
            # Train a new MLP classifier for each stock
            model = MLPClassifier(max_iter=1000, hidden_layer_sizes=hidden_layers, random_state=42)
            model_res, columns_res = train_model(model, train, stock)
            # Pickle the model object and save down
            with open(model_dir + stock + '_model.pkl', 'wb') as f: pickle.dump(model_res, f)
            with open(model_dir + stock + '_columns.pkl', 'wb') as f: pickle.dump(columns_res, f)
        end_time = time.time()
        # Report the training time
        print('Process execution took %s seconds' % (end_time - start_time))

    # For each stock in the portfolio, we run the prediction
    # We then run each of the reporting or explain processes as set in the environment variables
    for stock in core_stocks:
        # Load in the model
        with open(model_dir + stock + '_model.pkl', 'rb') as f: model_trained = pickle.load(f)
        with open(model_dir + stock + '_columns.pkl', 'rb') as f: model_columns = pickle.load(f)
        # Run the prediction
        run_prediction(model_trained, model_columns, train, test, stock)
        forecast = int(test[stock + '_model_position'].tail(1).iloc[0])

        if os.getenv('DAILY') == 'True':
            print('Generating daily report for ' + stock)
            reporting.daily_report(date_t, stock, date_t_data, date_tmin1_data, 'INPUT', date_t, 1, [], [], [], forecast)
        if os.getenv('SENTIMENT') == 'True':
            print('Generating sentiment report for ' + stock)
            trimmed_explain_date = date_t.replace('-', '')
            reporting.sentiment_report(stock, trimmed_explain_date)
        if os.getenv('HISTORY') == 'True':
            print('Generating trade history for ' + stock)
            reporting.history_report(test, stock, date_t)
        if os.getenv('ANCHORS') == 'True':
            print('Generating anchors report for ' + stock)
            explain_ANCHORS(date_t, stock, model_trained, test, model_columns, date_t_data, forecast)
        if os.getenv('SHAP') == 'True':
            print('Generating feature importance report for ' + stock)
            explain_SHAP(stock, date_t, model_trained, train, test, model_columns)
        if os.getenv('SIMILARITY') == 'True':
            print('Generating similarity report for ' + stock)
            explain_Similarity(stock, train, test, date_t_data, model_columns, forecast)
        if os.getenv('DICE') == 'True':
            print('Generating counterfactuals for ' + stock)
            explain_DICE(date_t, stock, model_trained, train, model_columns, date_t_data, forecast)
        if os.getenv('MODEL_PERF') == 'True':
            print('Generating report on model performance')
            reporting.performance_report(model_trained, train, test, model_columns, stock, date_t)

    if os.getenv('PORTFOLIO') == 'True':
        print('Running portfolio optimisation')
        portfolio_opt(test, core_stocks, date_t, date_tmin1)

    if os.getenv('CHALLENGER') == 'True':
        print('Running challenger model training and testing')
        run_challenger_model(train, test, core_stocks, date_t)

    if os.getenv('ALTERNATIVE') == 'True':
        print('Running alternative model training and testing')
        reporting.alt_models_report(train, test, core_stocks, date_t)

if __name__ == "__main__":
    main()
