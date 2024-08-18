import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Callable
from random import randint
from . import AdaptiveCP


class ACP_plots:

    @staticmethod
    def one_plot(data, y):
        '''
        Plot various visualizations for Adaptive Conformal Prediction.

        Parameters:
        - data (dict): A dictionary containing the following keys:
            - 'error_t_list' (list): List of error values.
            - 'conformal_sets' (list): List of conformal sets.
            - 'realised_interval_coverage' (float): Realized interval coverage.
            - 'interval_size' (int): Size of the interval.
            - 'alpha_t_list' (list, optional): List of alpha values. Default is None.
            - 'B_t_list' (list, optional): List of B values. Default is None.
            - 'model' (str): Name of the model.

        - y (list): List of y values.

        Returns:
        - None

        '''

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
        '''
        Plots various metrics related to adaptive conformal prediction.

        Parameters:
        - data (dict): A dictionary containing the following keys:
            - 'error_t_list' (list): List of error values.
            - 'conformal_sets' (list): List of conformal sets.
            - 'realised_interval_coverage' (float): Realized interval coverage.
            - 'interval_size' (int): Interval size.
            - 'interval_list' (list): List of interval values.
            - 'scale_list' (list): List of scale values.
            - 'model' (str): Name of the model.

        Raises:
        - ValueError: If 'scale_list' or 'interval_list' is not present in the data.

        Returns:
        - None: This function does not return anything. It only plots the metrics.
        '''

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
        '''
        Compare two methods and plot the results.

        Parameters:
        - method1 (dict): Dictionary containing information about the first method.
        - method2 (dict): Dictionary containing information about the second method.
        - figsize (tuple[int], optional): Figure size for the subplots. Default is (10, 5).

        Returns:
        - None

        '''

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
        '''
        Compare multiple methods and plot the results.
        
        Parameters:
        - list_of_methods (list[dict]): List of dictionaries containing information about the methods.
        - figsize (tuple[int], optional): Figure size for the subplots. Default is (10, 5).

        '''
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

    @staticmethod
    def AwDT_plot(data, dist: tuple, comparison = False):
        '''
        Plot various metrics related to Aw methods specificaly.
        
        Parameters:
        - data (dict): A dictionary containing the following
            - 'error_t_list' (list): List of error values.
            - 'conformal_sets' (list): List of conformal sets.
            - 'realised_interval_coverage' (float): Realized interval coverage.
            - 'interval_size' (int): Size of the interval.
            - 'interval_candidates' (list): List of interval candidates.
            - 'average_prediction_interval' (float): Average prediction interval.
            - 'chosen_interval_index' (list): List of chosen interval indices.
            - 'start_point' (int): Start point of the interval.
            - 'coverage_target' (float): Target coverage.
        - dist (tuple): Tuple containing the distribution.
        - comparison (bool, optional): Whether to compare with other methods. Default is False.
        '''

        y = dist[1]
        error_list = data['error_t_list']
        conformal_sets = data['conformal_sets']
        coverage = data['realised_interval_coverage']
        interval_size = data['interval_size']
        interval_candidates = data['interval_candidates']
        average_prediction_interval = data['average_prediction_interval']

        chosen_interval_index  = data['chosen_interval_index']
        start_point = data['start_point']

        if comparison:
            ACP = AdaptiveCP(data['coverage_target'])
            Dt_data = ACP.DtACI(dist, custom_interval=interval_candidates[len(interval_candidates)//2])
            ACI_data = ACP.ACI(dist, 0.05)

        fig = plt.figure(figsize=(30, 20))
        gs = gridspec.GridSpec(5, 2, figure=fig)

        # Realised Coverage
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(1 - pd.Series(error_list).rolling(interval_size).mean(), label='AwAci', color='b')
        ax1.axhline(coverage, color='b', linestyle='--')
        ax1.set_xlim(min(interval_candidates), len(y) - start_point)
        ax1.set_title('Realised Coverage')

        # Distance between upper and lower bounds
        ax2 = fig.add_subplot(gs[1, :])
        diff = [ele[1]-ele[0] for ele in conformal_sets]
        ax2.plot(diff, label='Distance')
        ax2.axhline(np.mean(diff), color='r', linestyle='--')
        ax2.set_xlim(min(interval_candidates), len(y) - start_point)
        ax2.set_ylim(bottom=0)
        ax2.set_title('Distance between upper and lower bounds')

        # Conformal sets
        ax3 = fig.add_subplot(gs[2:4, :])
        lower, upper = [ele[0] for ele in conformal_sets], [ele[1] for ele in conformal_sets]
        ax3.plot(lower, label='Lower')
        ax3.plot(upper, label='Upper')
        ax3.plot(y[start_point:])
        ax3.set_title('Conformal sets')
        ax3.set_ylim(bottom=np.percentile(lower, 2.5), top=np.percentile(upper, 97.5))
        ax3.set_xlim(0, len(y) - start_point)

        ax4 = fig.add_subplot(gs[4,0])
        ax4.plot(chosen_interval_index,label='chosen_interval_index')
        ax4.set_title('Chosen Interval')

        ax5 = fig.add_subplot(gs[4,1])
        ax5.axhline(average_prediction_interval, label='AwAci')

        if comparison:
            # Realised Coverage comparison
            comparison_error_list = Dt_data['error_t_list']
            comparison_coverage = Dt_data['realised_interval_coverage']
            comparison_prediciton_interval = Dt_data['average_prediction_interval']

            ax1.plot(1 - pd.Series(comparison_error_list).rolling(interval_size).mean(), linestyle='--', label='DtAci')
            ax1.axhline(comparison_coverage, color='g', linestyle=':')

            # Distance between upper and lower bounds comparison
            comparison_conformal_sets = Dt_data['conformal_sets']
            comparison_diff = [ele[1]-ele[0] for ele in comparison_conformal_sets]
            ax2.plot(comparison_diff, label='Comparison Distance', linestyle='--')
            ax2.axhline(np.mean(comparison_diff), color='g', linestyle=':', label='comparison average' )

            # Conformal sets comparison
            comparison_lower, comparison_upper = [ele[0] for ele in comparison_conformal_sets], [ele[1] for ele in comparison_conformal_sets]
            ax3.plot(comparison_lower, label='Comparison Lower', linestyle='--')
            ax3.plot(comparison_upper, label='Comparison Upper', linestyle='--')

            ax5.axhline(comparison_prediciton_interval, color='g', label='DtACI')

            aci_error_list = ACI_data['error_t_list']
            aci_coverage = ACI_data['realised_interval_coverage']
            aci_prediction_interval = ACI_data['average_prediction_interval']
            
            ax1.plot(1 - pd.Series(aci_error_list).rolling(interval_size).mean(), linestyle='-.', label='ACI Error')
            ax1.axhline(aci_coverage, color='r', linestyle='-.', label='ACI Coverage')
            
            ax5.axhline(aci_prediction_interval, color='r', label='ACI Prediction Interval')

        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()
        ax5.legend()

        plt.show()

    @staticmethod
    def analyse_MACI(data: tuple, method: Callable, candidates: list, shift_list: list[int], nu_sigma: tuple =(10**-2, 0.05), k: int=5, gamma=0.005):
        '''
        Analyse the results of MACI method.

        Parameters:
        - data (list): List of data points.
        - method (Callable): Method to be used.
        - candidates (list): List of interval candidates.
        - shift_list (list): List of shift points.
        - nu_sigma (tuple, optional): Tuple containing the nu and sigma values. Default is (10**-2, 0.05).
        - k (int, optional): Number of heads. Default is 5.
        - gamma (float, optional): Gamma value. Default is 0.005.

        Returns:
        - None
        '''
    
        result = method(data, interval_candidates=candidates, nu_sigma=nu_sigma, k=k, gamma=gamma)
        interval_candidates = result['interval_candidates']
        print(result['realised_interval_coverage'])

        plt.figure(figsize=(30, 15))
        plt.subplot(2, 2, 1)
        plt.plot(data[0])
        for shift in shift_list:
            plt.axvline(x=shift, color='red')
        plt.title('Data with Shift Points')

        shift_list = [x - max(interval_candidates) for x in shift_list]
        plt.subplot(2, 2, 2)
        rel_weights = np.column_stack(result['all_weights'])
        for i, x in enumerate(rel_weights):
            plt.plot(x, label=interval_candidates[i])
        for shift in shift_list:
            plt.axvline(x=shift, color='red')
        plt.legend()
        plt.title('Weight Distribution')

        # Above is good.

        all_eligible_heads_splits = np.split(result['eligible_heads_list'], shift_list)
        all_eligible_weights_splits = np.split(result['relative_eligible_final_weight_list'], shift_list)

        all_weights_splits = np.split(result['all_weights'], shift_list)
        all_radii_splits = np.split(result['radii_list'], shift_list)

        len_labels = [str(x) for x in interval_candidates]

        flattened_eligible_heads_splits = [x.flatten() for x in all_eligible_heads_splits]
        eligible_head_count = [np.unique(x, return_counts=True) for x in flattened_eligible_heads_splits]

        list_weight_dstack = [np.dstack(x) for x in all_weights_splits]
        list_mean_weight = [x.mean(axis=2).mean(axis=0) for x in list_weight_dstack]

        list_radii_dstack = [np.dstack(x) for x in all_radii_splits]
        list_mean_radii = [x.mean(axis=2).mean(axis=0) for x in list_radii_dstack]

        plt.subplot(2, 2, 3)
        plt.bar([str(x) for x in interval_candidates] + ['method'] , np.append(np.column_stack(result['radii_list']).mean(axis=1), result['average_prediction_interval']/2))
        plt.title('Radii Distribution')

        plt.tight_layout()
        plt.show()


        for i, (weight, radii) in enumerate(zip(list_mean_weight, list_mean_radii)):
            print('Section:', i, '-'*100)
            print(len(all_weights_splits[i]))
            
            _, axs = plt.subplots(2, 2, figsize=(20, 10))

            axs[0][0].pie(weight, labels=len_labels, autopct='%1.1f%%', startangle=140)
            axs[0][0].set_title('Weight Distribution')

            axs[0][1].bar(len_labels, radii)
            axs[0][1].set_title('Radii Distribution')

            heads, weights = eligible_head_count[i]
            hw_dict = dict(zip(heads, weights/sum(weights)))

            eligible_weighted = [(h, weight[h] * hw_dict[h]) if h in hw_dict else (h,0) for h in heads]
            print(eligible_weighted)

            axs[1][0].pie(weights, labels=[interval_candidates[w] for w in heads], autopct='%1.1f%%', startangle=140)
            axs[1][0].set_title('Relative Occurence of head which are elligible')

            axs[1][1].bar([str(interval_candidates[x[0]]) for x in eligible_weighted], [x[1] for x in eligible_weighted]) 
            axs[1][1].set_title('Weighted Occurence of head which are elligible')

            plt.show()

