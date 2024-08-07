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
    def no_shift(norm_dist: tuple = (0,1), seq_length: int = 500, data_transformation: Callable = None, datapoints: int = 1) -> list[tuple]:
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
    def single_shift(inital_dist: tuple = (0,1), shifted_dist: tuple = (1,1), seq_length: int = 500, shift_point: int = 50, data_transformation: Callable = None, datapoints: int = 1) -> list[tuple]:
        assert shift_point < seq_length, 'Shift point must be less then the sequence length.'
        
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
        all_price_data = tickers.history(period='5y', interval='1d')
        price_df = all_price_data[['close']].copy()
        stock_data_tuples = []

        # Some tickers in the list are incorrect or not trading so need 
        for ticker_symbol in price_df.index.get_level_values(0).unique():
            ticker_price_data = price_df.loc[ticker_symbol]
            ticker_close = ticker_price_data['close'].to_numpy()

            # Appending it to the stock_data_tuples list, the last volatilty is used as the prediciton for the next.
            stock_data_tuples.append((ticker_close[:-1], ticker_close[1:]))
        

        return stock_data_tuples
