import numpy as np
import pandas as pd
import os
from typing import Callable
from random import randint
import yahooquery as yq
import warnings
# Unfortunatley yahooquery produces a lot of warnings.
warnings.filterwarnings('ignore')

class ACP_data:
    @staticmethod
    def no_shift(norm_dist: tuple = (0,1), seq_length: int = 2000, data_transformation: Callable = None, datapoints: int = 1) -> list[tuple]:
        """
        Generate a list of tuples containing input and label data pairs.

        Args:
            norm_dist (tuple, optional): A tuple representing the mean and standard deviation of the normal distribution. Defaults to (0, 1).
            seq_length (int, optional): The length of the generated sequence. Defaults to 2000.
            data_transformation (Callable, optional): A function that transforms the generated data. Defaults to None.
            datapoints (int, optional): The number of data points to generate. Defaults to 1.

        Returns:
            list[tuple]: A list of tuples, where each tuple contains the input data and corresponding label data.
        """
        
        if data_transformation is None:
            data_transformation = lambda x: np.append(x[0], 0.1*x[:-1] + x[1:])

        all_label_value_pairs = []
        
        for _ in range(datapoints):
            norm_vals = np.random.normal(norm_dist[0], norm_dist[1], seq_length)
            transformed_data = data_transformation(norm_vals)

            input_data, labels_data = transformed_data[:-1], transformed_data[1:]
            all_label_value_pairs.append((input_data, labels_data))

        return all_label_value_pairs

    @staticmethod
    def single_shift(inital_dist: tuple = (0,1), shifted_dist: tuple = (1,1), seq_length: int = 2000, shift_point: int = 50, data_transformation: Callable = None, datapoints: int = 1) -> list[tuple]:
        """
        Generate a list of tuples containing input and label data pairs with a single shift point.

        Args:
            inital_dist (tuple, optional): Mean and standard deviation of the initial distribution. Defaults to (0,1).
            shifted_dist (tuple, optional): Mean and standard deviation of the shifted distribution. Defaults to (1,1).
            seq_length (int, optional): Length of the generated sequence. Defaults to 2000.
            shift_point (int, optional): Index at which the shift occurs in the sequence. Defaults to 50.
            data_transformation (Callable, optional): Function to transform the generated data. Defaults to None.
            datapoints (int, optional): Number of data points to generate. Defaults to 1.

        Returns:
            list[tuple]: List of tuples containing input and label data pairs.
        """
        
        assert shift_point < seq_length, 'Shift point must be less than the sequence length.'
        
        if data_transformation is None:
            data_transformation = lambda x: np.append(x[0], 0.1*x[:-1] + x[1:])

        all_label_value_pairs = []
      
        for _ in range(datapoints):
            norm_vals = np.random.normal(inital_dist[0], inital_dist[1], seq_length)
            
            norm_vals[shift_point:] = np.random.normal(shifted_dist[0], shifted_dist[1], seq_length - shift_point)
            transformed_data = data_transformation(norm_vals)

            input_data, labels_data = transformed_data[:-1], transformed_data[1:]
            all_label_value_pairs.append((input_data, labels_data))

        return all_label_value_pairs

    
    @staticmethod
    def multiple_shift(dist_shifts: list[tuple], seq_length : int = 500, shift_points: list[float] = None, data_transformation: Callable = None, datapoints: int = 1) -> list[tuple]:
        '''shift_points: should be an ordered set of proportions with length one less than the length of dist_shifts.'''
        
        if data_transformation is None:
            data_transformation = lambda x: np.append(x[0], 0.1*x[:-1] + x[1:])

        if shift_points is None:
            shift_points = np.array([1/len(dist_shifts) * i for i in range(1, len(dist_shifts))])

        assert np.all(np.diff(shift_points) > 0), 'Shift points must be in increasing order.'
        assert len(dist_shifts) == len(shift_points) + 1, 'The number of shifts must be one less than the number of shift points.'

        shift_points = np.round(np.array(shift_points) *seq_length ).astype(int)

        all_label_value_pairs = []

        for _ in range(datapoints):
            #first shift 
            norm_vals = np.random.normal(dist_shifts[0][0], dist_shifts[0][1], shift_points[0])

            for i, x in enumerate(np.diff(shift_points)):
                norm_vals = np.concatenate((norm_vals, np.random.normal(dist_shifts[i+1][0], dist_shifts[i+1][1], x)))

            # last shift
            norm_vals = np.concatenate((norm_vals, np.random.normal(dist_shifts[-1][0], dist_shifts[-1][1], seq_length - shift_points[-1])))
            
            transformed_data = data_transformation(norm_vals)
            input_data, labels_data = transformed_data[:-1], transformed_data[1:]
            all_label_value_pairs.append((input_data, labels_data))

        return all_label_value_pairs
    
    @staticmethod
    def _random_dist(dist_range: tuple = ((-10, 1), (10, 10))) -> tuple[int]:
        mean = np.random.uniform(min(dist_range[0][0], dist_range[1][0]), max(dist_range[0][0], dist_range[1][0]))
        var = np.random.uniform(min(dist_range[0][1], dist_range[1][1]), max(dist_range[0][1], dist_range[1][1]))
        return mean, var
    
    @staticmethod
    def random_shift(datapoints: int = 500, seq_range: tuple[int] = (500, 1000), dist_range: tuple = ((-10, 1), (10, 10)), shift_range: tuple[float] = (0.3, 0.7), data_transformation: Callable = None) -> list[tuple]:  
        ''' dist_range, gives min and max mean, and min and max variance.
            shift_range, gives the min and max shift point as a proportion of the sequence length.
            seq_range, gives the min and max sequence length.'''
        
        if data_transformation is None:
            data_transformation = lambda x: np.append(x[0], 0.1*x[:-1] + x[1:])

        all_label_value_pairs = []

        for _ in range(datapoints):
            seq_length = randint(seq_range[0], seq_range[1])
            shift_point = randint(int(shift_range[0]*seq_length), int(shift_range[1]*seq_length))
            Idist, Fdist = ACP_data._random_dist(dist_range), ACP_data._random_dist(dist_range)

            all_label_value_pairs.extend(ACP_data.single_shift(inital_dist=Idist, shifted_dist=Fdist, seq_length=seq_length, shift_point=shift_point, data_transformation=data_transformation, datapoints=1))
        
        return all_label_value_pairs
    
    @staticmethod
    def random_multi_shift(datapoints: int = 500, seq_range: tuple[int] = (1000, 2000), dist_range: tuple = ((-10, 1), (10, 10)), number_shift_range: tuple[float] = (2, 5), data_transformation: Callable = None) -> list[tuple]:
        ''' dist_range, gives min and max mean, and min and max variance.
            number_shift_range, gives the min and max number of shifts as a proportion of the sequence length.
            seq_range, gives the min and max sequence length.'''
        
        assert 1 <= number_shift_range[0] < number_shift_range[1], 'Number of shifts must be greater than 1 and the first number must be less than the second.'
        
        if data_transformation is None:
            data_transformation = lambda x: np.append(x[0], 0.1*x[:-1] + x[1:])

        all_label_value_pairs = []

        for _ in range(datapoints):
            seq_length = randint(seq_range[0], seq_range[1])
            number_shifts = randint(number_shift_range[0], number_shift_range[1])
            
            shift_points = np.sort([np.random.uniform(0.1, 1) for _ in range(number_shifts - 1)])
        
            dist_shifts = [ACP_data._random_dist(dist_range) for _ in range(number_shifts)]

            all_label_value_pairs.extend(ACP_data.multiple_shift(dist_shifts=dist_shifts, seq_length=seq_length, shift_points=shift_points, data_transformation=data_transformation, datapoints=1))
        
        return all_label_value_pairs  
    
    @staticmethod
    def stock_data(datapoints: int = 100, slc: slice = None) -> tuple[int]:
        '''Returns a list of tuples containing the stock data for the given number of datapoints.'''

        # Open the file using the relative path
        with open( r'C:\Users\tobyw\Documents\ChrisPython\ConformalProject\scripts\snptickers.txt', 'r') as f:
            all_tickers = f.read().splitlines()
            all_tickers.sort()

        if slc is None:        
            stock_tickers = all_tickers[:datapoints]
        else:
            stock_tickers = all_tickers[slc]

        tickers = yq.Ticker(stock_tickers)
        all_price_data = tickers.history(period='10y', interval='1d')
        price_df = all_price_data[['close']].copy()
        stock_data_tuples = []

        # Some tickers in the list are incorrect or not trading so need 
        for ticker_symbol in price_df.index.get_level_values(0).unique():
            ticker_price_data = price_df.loc[ticker_symbol]
            ticker_close = ticker_price_data['close'].to_numpy()

            # Appending it to the stock_data_tuples list, the last volatilty is used as the prediciton for the next.
            stock_data_tuples.append((ticker_close[:-1], ticker_close[1:]))
        

        return stock_data_tuples
    
    @staticmethod
    def test_on_stock_data(ACP_instance, ACP_method, datapoints: int, withvar: bool = False,  *args):
        '''Run method on multiple stock distributions and then return dictionary of results.'''
        
        results_dict = {'coverge':[], 
                        'width':[],
                        'raw_results': []}
        
        stock_data = ACP_data.stock_data(datapoints)
        
        if withvar:
            stock_data = ACP_data.xvy_from_ACP(stock_data)
        else:
            stock_data = ACP_data.xvy_correction(stock_data)
        

        for data in stock_data:
            
            result = ACP_method(ACP_instance, data, *args)

            results_dict['raw_results'].append(result)
            results_dict['coverge'].append(result['realised_interval_coverage'])
            results_dict['width'].append(result['average_prediction_interval'])

        
        coverage_mean = np.mean(results_dict['coverge'])
        width_mean = np.mean(results_dict['width'])

        results_dict['coverage_mean'] = coverage_mean
        results_dict['width_mean'] = width_mean

        return results_dict
    
    @staticmethod
    def create_nomrmal_cheb_data(length: int, model_error: float = 0.1, time_series_function: callable = lambda x: x, var_range: tuple = (0.5, 2)) -> tuple:
        minv, maxv = var_range

        true_variance_array = np.random.uniform(minv, maxv, length)
        corresponding_normal = np.random.normal(0, true_variance_array, length)

        model_variance_array = true_variance_array + np.random.uniform(model_error*minv, model_error*maxv, length)
        time_series_normal = time_series_function(corresponding_normal)

        # Now we need to return as xpred, varpred, y
        # The variance is the prediction for the same time step. 
        # Hence you need to ignore the first value for the variance as you do for the true value.
        
        return (time_series_normal[:-1], model_variance_array[1:], time_series_normal[1:])

    @staticmethod
    def create_hetroskedatic_cheb_data(length: int, model_error: float = 0.1, var_range: tuple = (0.5, 2)) -> tuple:
        '''Model error roughly corresponds to percentage uncertainty in the model.'''  

        # We will simulate a random walk for the variance. Might do a exponenital random walk as 
        # then no issues with negative values.

        exp_random_walk = np.random.normal(0, 0.1, length)
        true_variance_array = np.exp(np.cumsum(exp_random_walk))

        corresponding_normal = np.random.normal(0, true_variance_array, length)

        model_variance_array = true_variance_array + (true_variance_array * np.random.uniform(model_error, 2-model_error, length))
        time_series_normal = np.cumsum(corresponding_normal)

        # Now we need to return as xpred, varpred, y

        return (time_series_normal[:-1], model_variance_array[1:], time_series_normal[1:])

    @staticmethod
    def xvy_from_y(series, lookback: int = -1):
        '''This function will create the x, var and y series from the y series.'''
        
        if lookback == -1:
            lookback = len(series) # This results in all data being used.

        x = series[:-1]
        y = series[1:]

        # Calculating the series of sample variances of a length lookback.
        var = [np.var(series[max(0, i - lookback):i+1]) for i in range(1, len(series))]

        return x, var, y

    @staticmethod
    def xvy_from_ACP(dataset, lookback: int = -1):
        ''' Converts data from ACP format to xvy format.'''

        lookback = len(dataset) if lookback == -1 else lookback
        final = []
        for x, y in dataset:
            var = [np.var(x[max(0, i - lookback):i+1]) for i in range(1, len(x))] # We need to ignore the first value.
            final.append((x[1:], var, y[1:]))

        return final

    @staticmethod
    def xvy_correction(dataset):
        ''' Corrects ACP data so that it can be compared with xvy counterpart.'''
        return [(x[0][1:], x[1][1:]) for x in dataset]
    
    @staticmethod
    def simple_comparison(result_list_1: list, result_list_2: list):
        ''' Compares on the following metrics:
        Throughout, 0 corresponds to the first model and 1 corresponds to the second model.
        - relative_width: The ratio of the average prediction interval of the first model to the second model.
        - average_difference_coverage: The difference between the realised coverage and the target coverage.
        - better_coverage: A binary value of which model has the closer coverage to target.
        - largest_coverage_deviation: A binary value of which model has the largest deviation from the target coverage.
        - largest_ratio_above_and_below_deviation: A binary value of which model has the largest ratio of above and below the target coverage.'''

        raw_metric_dict = {
            'better_coverage:': [], # closer is better
            'relative_width': [],
            'average_difference_coverage': [],
            'relative_width_no_outliers': [],
            'average_coverage_difference': [],
            'largest_coverage_deviation': [], # closer is worse
            'largest_ratio_above_and_below_deviation': [], # closer is worse.
        }

        for r1, r2 in zip(result_list_1, result_list_2):
            # Relative width and average difference coverage.
            raw_metric_dict['relative_width'].append(r1['average_prediction_interval'] / r2['average_prediction_interval'])
            raw_metric_dict['average_difference_coverage'].append(r1['realised_interval_coverage'] - r2['realised_interval_coverage'])

            # better coverage.
            if abs(r1['realised_interval_coverage'] - 1 + r1['coverage_target']) < abs(r2['realised_interval_coverage'] - 1 + r2['coverage_target']):
                # r1 better
                raw_metric_dict['better_coverage:'].append(0)
            else:
                raw_metric_dict['better_coverage:'].append(1)

            # Largest deviation from desired coverage.
            r1_rolling_coverage_deviation = pd.Series(r1['error_t_list']).rolling(r1['interval_size']).mean() - r1['coverage_target']
            r2_rolling_coverage_deviation = pd.Series(r2['error_t_list']).rolling(r2['interval_size']).mean() - r2['coverage_target']

            if max(abs(r1_rolling_coverage_deviation.dropna())) > max(abs(r2_rolling_coverage_deviation.dropna())):
                # r1 bigger
                raw_metric_dict['largest_coverage_deviation'].append(0)
            else:
                raw_metric_dict['largest_coverage_deviation'].append(1)

            # Time spent below desired vs above.
            r1_above = r1_rolling_coverage_deviation < 0
            r2_above = r2_rolling_coverage_deviation < 0

            r1_ratio_above = sum(r1_above) / len(r1_above)
            r2_ratio_above = sum(r2_above) / len(r2_above)

            if abs(r1_ratio_above - 0.5) > abs(r2_ratio_above - 0.5):
                #r1 bigger
                raw_metric_dict['largest_ratio_above_and_below_deviation'].append(0)
            else:
                raw_metric_dict['largest_ratio_above_and_below_deviation'].append(1)
        
        # Removing outliers.
        q1 = np.percentile(raw_metric_dict['relative_width'], 5)
        q3 = np.percentile(raw_metric_dict['relative_width'], 95)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        np_rel = np.array(raw_metric_dict['relative_width'])
        relative_width_outliers_removed = np_rel[(np_rel >= lower_bound) & (np_rel <= upper_bound)]

        # taking averages of relevant keys and adding to the metric_dict.
        results_dict = {
            'relative_width_mean': np.mean(raw_metric_dict['relative_width']),
            'relative_width_std': np.std(raw_metric_dict['relative_width']),
            'relative_width_no_outliers_mean': np.mean(relative_width_outliers_removed),
            'average_coverage_difference_mean': np.mean(raw_metric_dict['average_difference_coverage']),
            'largest_coverage_deviation_mean': np.mean(raw_metric_dict['largest_coverage_deviation']),
            'largest_ratio_above_and_below_deviation_mean': np.mean(raw_metric_dict['largest_ratio_above_and_below_deviation']),
            'better_coverage_mean': np.mean(raw_metric_dict['better_coverage:'])
        }
    
        return results_dict, raw_metric_dict