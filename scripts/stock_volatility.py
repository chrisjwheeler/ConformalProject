import os
import sys
import argparse
import logging
import warnings
import pickle
import numpy as np
import pandas as pd
import yahooquery as yq

warnings.filterwarnings('ignore')


# Import the Conformal Prediction methods
src_path = os.path.abspath(os.path.join('..', 'src'))
sys.path.append(src_path)

from ConformalMethods import AdaptiveCP, ACP_plots

def save_checkpoint(checkpoint, filename):
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint, f)

def load_checkpoint(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

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
    price_df['volatility'] = price_df['close'].pct_change()**2 # this shouldnt be here shoudl be inside I dount would make a hige difference.

    logging.info('Described volatilty data' + price_df['volatility'].describe())

    # I have the volatilties, When are the score calcualted? they are calculated in the conformal prediction method.
    # So I just need to pass the volatilities to the conformal prediciton method.
    # Well it needs the models prediciton and the true value, so you can either use the last datapoint as the prediciton or you can use and ARIMA model as the prediction.

    # If you use the last prediction then, xpred = vol[:-1], ypred = vol[1:]

    stock_data_tuples = []

    for ticker_symbol in stock_tickers:
        # Getting the volatilty data for each ticker
        ticker_price_data = price_df.loc[ticker_symbol]
        ticker_volatilty = ticker_price_data['volatility'].to_numpy()

        # As when creating the volaility there is an intial NaN value, I will remove this.
        ticker_volatilty = ticker_volatilty[1:]

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
            #ACP_plots.one_plot(method_result, data[1])

        except KeyboardInterrupt:
            logging.info('Keyboard interrupt, saving checkpoint')
            save_checkpoint(conformal_data, 'checkpoint.pkl')
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

        # Save the checkpoint
        if i%10 == 0:
            save_checkpoint(conformal_data, method + '_checkpoint.pkl')

    logging.info('All data processed')

 
if __name__ == '__main__':

    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Test Conformal Prediction method on stock volatility')
    parser.add_argument('--alpha', type=float, help='Significance level for the conformal prediction', default=0.05)
    parser.add_argument('--method', choices=['ACI', 'DtACI', 'AwACI', 'AwDtACI'], help='Confomral Prediction method to use', required=True)
    parser.add_argument('--datapoints', type=int, help='Number of stocks to use', default=50)
    parser.add_argument('--resume', type=str, help='Checkpoint file to resume from', default=None)

    args=parser.parse_args()

    # Set up logging
    log_name = 'stock_volatility_' + args.method + '.log'
    logging.basicConfig(filename=log_name, level=logging.INFO, format='%(asctime)s:%(levelname)s - %(message)s')

    logging.info(' --------- Starting stock volatility prediction ---------')

    # Load the checkpoint
    if args.resume:
        conformal_data = load_checkpoint(args.resume)
        logging.info('Resuming from checkpoint')
    else:
        conformal_data = {
            'next': 0,
            'target': args.datapoints,
            'individual_results': []
            }
        logging.info('Created data dictionary', conformal_data)

    # Get the stock data
    stock_data = get_stock_data(conformal_data['next'], conformal_data['target'])

    # Run the conformal prediction
    run_conformal_prediction(conformal_data, stock_data, args.alpha, args.method)

    # Save the results
    
    # Compute averages of each key in the individual results
    averages = {}
    for key in conformal_data['individual_results'][0].keys():
        averages[key] = np.mean([x[key] for x in conformal_data['individual_results']])

    # Save the averages to a csv file
    with open(args.method + '_results.csv', 'w') as f:
        f.write('Method,Alpha,Target,Current,')
        f.write(','.join(averages.keys()) + '\n')
        f.write(f'{args.method},{args.alpha},{conformal_data["target"]},{conformal_data["current"]},')
        f.write(','.join([str(x) for x in averages.values()]) + '\n')
    
    logging.info('Results saved')
        


data = {
    'current':0,
    'target': 50,
    'individual_results': [{'ticker': 'AAPL',
                 'coverage': 0.95,
                 'width': 10,
                 'Qwidth1': 0.1,
                 'Qwidth25': 0.25,
                 'Qwidth50': 0.5,
                 'Qwidth75': 0.75,
                 'Qwidth99': 0.99,
                 }]
}
