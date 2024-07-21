import os
import sys
import argparse
import logging
import warnings
import pickle
import numpy as np
import pandas as pd
import yahooquery as yq
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Import the Conformal Prediction methods
src_path = os.path.abspath(os.path.join('..', 'src'))
sys.path.append(src_path)

from ConformalMethods import AdaptiveCP, ACP_plots

def get_stock_data(start_index, end_index):
    # Open txt file containg ticker names
    with open(r'C:\Users\tobyw\Documents\ChrisPython\ConformalProject\scripts\snptickers.txt', 'r') as f:
        all_tickers = f.read().splitlines()
        all_tickers.sort()
    
    stock_tickers = all_tickers[start_index:end_index]

    tickers = yq.Ticker(stock_tickers)
    all_price_data = tickers.history(period='5y', interval='1d')
    price_df = all_price_data[['close']].copy()
    
    # Calculate the volatility
    price_df['volatility'] = price_df['close'].pct_change()**2

    stock_data_tuples = []

    for ticker_symbol in stock_tickers:
        # Getting the volatilty data for each ticker
        ticker_price_data = price_df.loc[ticker_symbol]
        ticker_volatilty = ticker_price_data['volatility'].to_numpy()

        # As when creating the volaility there is an intial NaN value, I will remove this.
        ticker_volatilty = ticker_volatilty[1:]

        #plt.plot(ticker_volatilty[350:550])
        #plt.show()

        contains_nan = np.isnan(ticker_volatilty).any()
        if contains_nan:
            logging.warning(f'{ticker_symbol} contains NaN values')
        else:
            logging.info(f'{ticker_symbol} loaded successfully')

        # Appending it to the stock_data_tuples list, the last volatilty is used as the prediciton for the next.
        stock_data_tuples.append((ticker_symbol, (ticker_volatilty[:-1], ticker_volatilty[1:])))

    assert len(stock_data_tuples) == len(stock_tickers)
    
    logging.info(f'Loaded data for {len(stock_data_tuples)} stocks')

    return stock_data_tuples

def run_conformal_prediction(conformal_data: dict, stock_data: list, alpha: float, method: str):
    logging.info(f'Running conformal prediction with alpha={alpha} and method={method}')

    # intialise Adative CP
    ACP = AdaptiveCP(alpha, 100)

    method_dict = {
        'ACI': ACP.ACI,
        'DtACI': ACP.DtACI,
        'AwACI': ACP.AwACI,
        'AwDtACI': ACP.AwDtACI
    }

    CP_method = method_dict[method]

    for i, (ticker_symbol, data) in enumerate(stock_data, start=conformal_data['next']):
        # Run the conformal prediction method on the data

        # Get the result dictionary, excepting exceptions and continuing unless there is a keyboard interrupt at which point you save the current state and raise the exception.
        try:
            method_result = CP_method(data)
            #ACP_plots.plot_alpha_t(method_result)
            plt.plot(method_result['alpha_t_list'])
            plt.show()
            ACP_plots.one_plot(method_result, data[1])

        except KeyboardInterrupt:
            logging.info('Keyboard interrupt, saving checkpoint')
            raise
        
        except Exception as e:
            logging.error(f'Error in {ticker_symbol}: {e}')
            continue

        # Calculate the coverage and width of the intervals
        average_coverage = method_result['realised_interval_coverage']
        average_prediction_interval = method_result['average_prediction_interval']

        individual_prediction_widths = list(map(lambda x: x[1] - x[0], method_result['conformal_sets']))


        # Calculate the quantiles of the width
        Qwidth1 = np.quantile(individual_prediction_widths, 0.01)
        Qwidth25 = np.quantile(individual_prediction_widths, 0.25)
        Qwidth50 = np.quantile(individual_prediction_widths, 0.5)
        Qwidth75 = np.quantile(individual_prediction_widths, 0.75)
        Qwidth99 = np.quantile(individual_prediction_widths, 0.99)

        # Add the results to the output
        conformal_data['individual_results'].append({
            'ticker': ticker_symbol,
            'coverage': average_coverage,
            'width': average_prediction_interval,
            'Qwidth1': Qwidth1,
            'Qwidth25': Qwidth25,
            'Qwidth50': Qwidth50,
            'Qwidth75': Qwidth75,
            'Qwidth99': Qwidth99
        })

        # Log the point
        logging.info(f'{i} - {ticker_symbol} - Coverage: {average_coverage}, Width: {average_prediction_interval}, Qwidth1: {Qwidth1}, Qwidth25: {Qwidth25}, Qwidth50: {Qwidth50}, Qwidth75: {Qwidth75}, Qwidth99: {Qwidth99}')

        # Update the current index
        conformal_data['next'] = i


alpha = 0.1
method = 'ACI'
datapoints = 5

logging.info('All data processed')

# Set up logging
log_name = 'stock_volatility_debug_' + method + '.log'
logging.basicConfig(filename=log_name, level=logging.INFO, format='%(asctime)s:%(levelname)s - %(message)s')

logging.info(' --------- Starting stock volatility prediction ---------')

conformal_data = {
    'next': 0,
    'target': datapoints,
    'individual_results': []
    }
logging.info('Created data dictionary', conformal_data)

# Get the stock data
stock_data = get_stock_data(conformal_data['next'], conformal_data['target'])

# Run the conformal prediction
run_conformal_prediction(conformal_data, stock_data, alpha, method)

    # Save the results
    