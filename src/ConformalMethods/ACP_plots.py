import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Callable
from random import randint

class ACP_plots:
    def __init__(self):
        pass
    
    @staticmethod
    def plot_conformal_intervals(data_list: list):
        _, ax = plt.subplots()
        for data in data_list:
            ax.plot(data['conformal_sets'], label=data['model'])
            ax.plot(data['error_t'])
        
        ax.legend()
        ax.set_title('Conformal Intervals')
        ax.set_xlabel('Time')
        ax.set_ylabel('Conformal Interval')
        plt.show()

    @staticmethod
    def plot_alpha_t(data_list: list):
        _, ax = plt.subplots()
        for data in data_list:
            if 'alpha_t_list' in data:
                ax.plot(data['alpha_t_list'], label=data['model'])

        ax.legend()
        ax.set_title('Alpha_t')
        ax.set_xlabel('Time')
        ax.set_ylabel('Alpha_t')
        plt.show()

    @staticmethod
    def plot_y(data: list[tuple], y: list, gap: int = 0, figsize: tuple[int] =(30, 15)) -> None:
        interval_size = max(51, gap+1)
        bottom = [ele[0] for ele in data['conformal_sets']]
        top = [ele[1] for ele in data['conformal_sets']]
        print(len(y), len(bottom), len(top))

        _, ax = plt.subplots(figsize=figsize)
        ax.plot(y[interval_size:], label='True Value')
        ax.plot(bottom, label='Lower Bound')
        ax.plot(top, label='Upper Bound')

        ax.legend()
        ax.set_title('Conformal Prediction')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        plt.show()


    @staticmethod
    def one_plot(data, y):

        error_list = data['error_t_list']
        conformal_sets = data['conformal_sets']
        coverage = data['realised_interval_coverage']
        interval_size = data['interval_size']

        fourthplot = False
        if 'alpha_t_list' in data and 'B_t_list' in data:
            alpha_list = data['alpha_t_list']
            B_t_list = data['B_t_list']
            fourthplot = True

        fig, axs = plt.subplots(2, 2, figsize=(20, 10))
        fig.suptitle('Adaptive Conformal Prediction for '+ data['model'])

        axs[0][0].plot(1 - pd.Series(error_list).rolling(interval_size).mean())
        axs[0][0].axhline(coverage, color='r', linestyle='--')
        axs[0][0].set_title('Realised Coverage')
        
        axs[0][1].plot([ele[0] for ele in conformal_sets], label='Lower')
        axs[0][1].plot([ele[1] for ele in conformal_sets], label='Upper')
        axs[0][1].plot(y[interval_size+1:])
        axs[0][1].set_title('Conformal sets')
        axs[0][1].legend()

        axs[1][0].plot([ele[1]-ele[0] for ele in conformal_sets], label='Distance')
        axs[1][0].axhline(np.mean([ele[1]-ele[0] for ele in conformal_sets]), color='r', linestyle='--')
        axs[1][0].legend()
        axs[1][0].set_title('Distance between upper and lower bounds')

        if fourthplot:
            axs[1][1].plot(alpha_list,label='our alpha')
            axs[1][1].plot(B_t_list, label='alpha*')
            axs[1][1].legend()
            axs[1][1].set_title('Alpha_t and B_t')

        plt.show()

    @staticmethod
    def plot_scale_interval(data):
        ## For experiments with scale and interval size

        if 'scale_list' not in data or 'interval_list' not in data:
            raise ValueError('Data does not contain scale_list or interval_list')

        error_list = data['error_t_list']
        conformal_sets = data['conformal_sets']
        coverage = data['realised_interval_coverage']
        interval_size = data['interval_size']
        interval_list = data['interval_list']
        scale_list = data['scale_list']

        fig, axs = plt.subplots(3, 2, figsize=(15, 7))
        fig.suptitle('Adaptive Conformal Prediction for '+ data['model'])

        axs[0][0].plot(1 - pd.Series(error_list).rolling(interval_size).mean())
        axs[0][0].axhline(coverage, color='r', linestyle='--')
        axs[0][0].set_title('Realised Coverage')
        
        axs[0][1].plot([ele[0] for ele in conformal_sets], label='Lower')
        axs[0][1].plot([ele[1] for ele in conformal_sets], label='Upper')
        axs[0][1].set_title('Conformal sets')
        axs[0][1].legend()

        axs[1][0].plot([ele[1]-ele[0] for ele in conformal_sets], label='Distance')
        axs[1][0].axhline(np.mean([ele[1]-ele[0] for ele in conformal_sets]), color='r', linestyle='--')
        axs[1][0].legend()
        axs[1][0].set_title('Distance between upper and lower bounds')

        axs[1][1].plot(interval_list, label='lookback')
        axs[1][1].axhline(np.mean(interval_list), color='r', linestyle='--')
        axs[1][1].axhline(interval_size, color='g', linestyle='--')
        axs[1][1].legend()
        axs[1][1].set_title('Interval size')

        axs[2][0].plot(scale_list, label='scale')
        axs[2][0].axhline(np.mean(scale_list), color='r', linestyle='--')
        axs[2][0].legend()
        axs[2][0].set_title('Scale')

        plt.show()
    
    @staticmethod
    def compare_two(method1, method2, figsize: tuple[int] =(10, 5)):
        model1 = method1['model']
        model2 = method2['model']

        interval_size = method1['interval_size']

        fig, axs = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Comparison between '+ model1 + ' and ' + model2)

        axs[0][0].plot(1 - pd.Series(method1['error_t_list']).rolling(interval_size).mean(), label=model1)
        axs[0][0].plot(1 - pd.Series(method2['error_t_list']).rolling(interval_size).mean(), label=model2)
        axs[0][0].axhline(method1['realised_interval_coverage'], color='g', linestyle='-.', label=model1+' average')
        axs[0][0].axhline(method2['realised_interval_coverage'], color='r', linestyle='-.', label=model2+' average')
        axs[0][0].axhline(1 - method1['coverage_target'], color='b', linestyle=':')
        axs[0][0].legend()
        axs[0][0].set_title('Coverage')

        alphalist = [(m['alpha_t_list'], m['model']) for m in [method1, method2] if 'alpha_t_list' in m]
        for alpha in alphalist:
            axs[0][1].plot(alpha[0], label=alpha[1])
        axs[0][1].legend()
        axs[0][1].set_title('Alpha t')

        method1_distance = np.array([ele[1]-ele[0] for ele in method1['conformal_sets']])
        method2_distance = np.array([ele[1]-ele[0] for ele in method2['conformal_sets']])

        axs[1][0].plot(method1_distance - method2_distance)
        axs[1][0].axhline(np.mean(method1_distance - method2_distance), color='r', linestyle='--', label='mean difference')
        axs[1][0].legend()
        axs[1][0].set_title('Interval widths:' + model1+ ' - '+ model2)

        axs[1][1].set_title('Average Prediction Interval')
        axs[1][1].axhline(method1['average_prediction_interval'], color='r', label=model1)
        axs[1][1].axhline(method2['average_prediction_interval'], label=model2)
        axs[1][1].legend()
        
        plt.show()

    @staticmethod
    def compare_many(list_of_methods, figsize: tuple[int] =(10, 5)):
        interval_size = list_of_methods[0]['interval_size']

        # Create a 2x2 grid
        gs = gridspec.GridSpec(3, 2)

        fig = plt.figure(figsize=figsize)
        fig.suptitle('Comparison between different methods')

        # Create the subplots
        ax1 = plt.subplot(gs[0, :])
        ax2 = plt.subplot(gs[1, 0])
        ax3 = plt.subplot(gs[1, 1])  
        ax4 = plt.subplot(gs[2, :])  

        colors = ['blue', 'orange', 'red', 'cyan', 'magenta']
        for i, method in enumerate(list_of_methods):
            cr = colors[i % len(colors)]

            ax1.plot(1 - pd.Series(method['error_t_list']).rolling(interval_size).mean(), color=cr,  label=method['model'])
            ax1.axhline(method['realised_interval_coverage'], linestyle='-.', color=cr)

            if 'alpha_t_list' in method:
                ax2.plot(method['alpha_t_list'], color=cr, label=method['model'])
                
            method_distance = np.array([ele[1]-ele[0] for ele in method['conformal_sets']])
            
            ax3.axhline(method['average_prediction_interval'], color=cr, label=method['model'])

            ax4.plot(method_distance, color=cr, label=method['model'])

        ax1.axhline(1 - method['coverage_target'], color='black', linestyle=':')
        ax1.legend()
        ax1.set_title('Coverage')

        ax2.legend()
        ax2.set_title('Alpha t')

        ax3.legend()
        ax3.set_title('Average Prediction Interval')

        ax4.legend()
        ax4.set_title('Interval widths')

        plt.show()