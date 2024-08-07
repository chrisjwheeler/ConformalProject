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

from ConformalMethods import AdaptiveCP, ACP_data

def get_normal_data(amount):
    new_transform = lambda x: np.append(x[:2], 0.5*x[:-2] + 0.5*x[1:-1] + x[2:])
    random_shift = ACP_data.random_shift(amount, seq_range=(3000, 3001), data_transformation=new_transform)

    # dump the data to a pickle file
    with open(f'data/normal_data_{amount}.pkl', 'wb') as f:
        pickle.dump(random_shift, f)

    return random_shift


def run_conformal_prediction(conformal_data: dict, normal_data: list, alpha: float, method: str, startpoint=950):
    logging.info(f'Running conformal prediction with alpha={alpha} and method={method}')

    # intialise Adative CP
    ACP = AdaptiveCP(alpha, 100)

    method_dict = {
        'ACI': ACP.ACI,
        'DtACI': ACP.DtACI,
        'AwACI': ACP.AwACI,
        'AwDtACI': ACP.AwDtACI,
        'MACI': ACP.MACI,
    }

    CP_method = method_dict[method]

    for i, data in enumerate(normal_data, start=conformal_data['next']):
        
        # Run the conformal prediction method on the data
        try:
            if method in ['ACI', 'DtACI']:
                method_result = CP_method(data, custom_interval=300, startpoint=startpoint)
            else:
                method_result = CP_method(data)
        
        except Exception as e:
            logging.error(f'Error in {i}: {e}')
            continue

        # Calculate the coverage and width of the intervals
        average_coverage = method_result['realised_interval_coverage']
        average_prediction_interval = method_result['average_prediction_interval']

        individual_prediction_widths = list(map(lambda x: x[1] - x[0], method_result['conformal_sets']))

        # Calculate the quantiles of the width
        Qwidth1 = np.nanquantile(individual_prediction_widths, 0.01)
        Qwidth25 = np.nanquantile(individual_prediction_widths, 0.25)
        Qwidth50 = np.nanquantile(individual_prediction_widths, 0.5)
        Qwidth75 = np.nanquantile(individual_prediction_widths, 0.75)
        Qwidth99 = np.nanquantile(individual_prediction_widths, 0.99)

        if average_coverage is np.nan:
            logging.warning(f'{i} coverage is NaN')
            continue

        # Add the results to the output
        conformal_data['individual_results'].append({
            'i': i,
            'coverage': average_coverage,
            'width': average_prediction_interval,
            'Qwidth1': Qwidth1,
            'Qwidth25': Qwidth25,
            'Qwidth50': Qwidth50,
            'Qwidth75': Qwidth75,
            'Qwidth99': Qwidth99
        })

        # Log the point

        # Update the current index
        conformal_data['next'] = i

        # Save the checkpoint
        if i%10 == 0:
            logging.info(f'{i} - Coverage: {average_coverage}, Width: {average_prediction_interval}, Qwidth1: {Qwidth1}, Qwidth25: {Qwidth25}, Qwidth50: {Qwidth50}, Qwidth75: {Qwidth75}, Qwidth99: {Qwidth99}')

    logging.info('All data processed')

 
if __name__ == '__main__':

    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Test Conformal Prediction method on stock volatility')
    parser.add_argument('--alpha', type=float, help='Significance level for the conformal prediction', default=0.05)
    parser.add_argument('--method', choices=['ACI', 'DtACI', 'AwACI', 'AwDtACI', 'MACI'], help='Conformal Prediction method to use', required=True)
    parser.add_argument('--datapoints', type=int, help='Number of stocks to use', default=50)
    parser.add_argument('--data', type=str, help='Checkpoint file to resume from', default=None)

    args=parser.parse_args()

    # Set up logging
    log_name = f'logs/normal_data_{args.method}.log'
    logging.basicConfig(filename=log_name, level=logging.INFO, format='%(asctime)s:%(levelname)s - %(message)s')

    logging.info(' --------- Starting stock price prediction ---------')

    # Load the checkpoint

    conformal_data = {
        'next': 0,
        'target': args.datapoints,
        'individual_results': []
        }
    logging.info('Created data dictionary', conformal_data)

    # Get the stock data
    if args.data is not None:
        with open(f'data/{args.data}', 'rb') as f:
            normal_data = pickle.load(f)
            logging.info('Loaded checkpoint data')
    else:
        normal_data = get_normal_data(args.datapoints)

    # Run the conformal prediction
    run_conformal_prediction(conformal_data, normal_data, args.alpha, args.method)

    # Save the results
    
    # Compute averages of each key in the individual results
    averages = {}
    for key in list(conformal_data['individual_results'][0].keys())[1:]:
        contains_nan = np.isnan([x[key] for x in conformal_data['individual_results']]).any()
        if contains_nan:
            logging.warning(f'{key} contains NaN values, at position {np.where(np.isnan([x[key] for x in conformal_data["individual_results"]]))}')

        averages[key] = np.nanmean([x[key] for x in conformal_data['individual_results']])

    # Save the averages to a csv file
    with open(f'results/{args.method}_{args.datapoints}_{args.alpha}_normal_results.csv', 'w') as f:
        f.write('Method, Alpha, Target, Current,')
        f.write(','.join(averages.keys()) + '\n')
        f.write(f'{args.method},{args.alpha},{conformal_data["target"]},{conformal_data["next"]},')
        f.write(','.join([str(x) for x in averages.values()]) + '\n')
    
    logging.info('Results saved')
        