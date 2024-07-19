import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from random import randint


class AdaptiveCP:
    def __init__(self, coverage_target=0.1, interval_size: int =50, score_function=None, neg_inverse_score=None, pos_inverse_score=None):
        self.coverage_target = coverage_target
        self.interval_size = interval_size
        self.score_function = score_function if score_function is not None else lambda xpred, y: (abs(y - xpred))/abs(xpred)
        self.neg_inverse_score = neg_inverse_score if neg_inverse_score is not None else lambda x_t, Q: (x_t) - (abs(x_t) * Q)
        self.pos_inverse_score = pos_inverse_score if pos_inverse_score is not None else lambda x_t, Q: (x_t) + (abs(x_t) * Q)

    def C_t(self, alpha_t, scores, x_t, t, custom_interval = None):
        # Insuring that alpha_t is between 0 and 1
        alpha_t = min(1, max(0, alpha_t))
        interval = custom_interval if custom_interval is not None else self.interval_size
        
        Q = np.quantile(scores[t-interval:t], 1 - alpha_t)
        positve_v = self.pos_inverse_score(x_t, Q)
        negative_v = self.neg_inverse_score(x_t, Q)
        
        return negative_v, positve_v
    
    @staticmethod
    def err_t(y_t, C_t_interval):
        if C_t_interval[0] < y_t < C_t_interval[1]:
            return 0
        else:
            return 1
        
    def l(self, B, theta):
        return (self.coverage_target * (B - theta)) - min(0, (B - theta))
    
    def vectorize_l(self):
        return np.vectorize(self.l)
        
    def NonAdaptive(self, timeseries_data: tuple, custom_interval: int = None) -> dict:
        xpred, y = timeseries_data
        alpha_t = self.coverage_target

        if custom_interval is not None:
            bigger_interval = max(custom_interval, self.interval_size) + 1
        else:
            bigger_interval = self.interval_size + 1
 
        All_scores = self.score_function(xpred, y)

        err_t_list = []
        conformal_sets_list = []
        
        for i in range(bigger_interval, len(All_scores)):
            Coverage_t = self.C_t(alpha_t=alpha_t, scores=All_scores, x_t=xpred[i], t=i, custom_interval=custom_interval)
            conformal_sets_list.append(Coverage_t)

            error_t = AdaptiveCP.err_t(y[i], Coverage_t)
            err_t_list.append(error_t)

        # Calculating different averages
        realised_interval_coverage = 1 - pd.Series(err_t_list).rolling(self.interval_size).mean().mean()
        average_prediction_interval = np.mean([abs(x[1] - x[0]) for x in conformal_sets_list])

        return {
            'model': 'Non Adaptive CP',
            'coverage_target': self.coverage_target,
            'realised_interval_coverage': realised_interval_coverage,
            'average_prediction_interval': average_prediction_interval,
            'conformal_sets': conformal_sets_list,
            'error_t_list': err_t_list, 
            'interval_size': self.interval_size
        }

    def ACI(self, timeseries_data: tuple, gamma: float, custom_interval = None) -> dict:
        xpred, y = timeseries_data
        alpha_t_list = [self.coverage_target]

        All_scores = self.score_function(xpred, y)

        err_t_list = []
        conformal_sets_list = []
        
        for i in range(self.interval_size+1, len(All_scores)):
            Coverage_t = self.C_t(alpha_t_list[-1], All_scores, xpred[i], i, custom_interval)
            conformal_sets_list.append(Coverage_t)

            error_t = AdaptiveCP.err_t(y[i], Coverage_t)
            err_t_list.append(error_t)

            alpha_t = min(max(alpha_t_list[-1] + (gamma * (self.coverage_target - error_t)), 0), 1)
            alpha_t_list.append(alpha_t)

        # Calculating different averages
        realised_interval_coverage = 1 - pd.Series(err_t_list).rolling(self.interval_size).mean().mean()
        average_prediction_interval = np.mean([abs(x[1] - x[0]) for x in conformal_sets_list])

        return {
            'model': 'ACI',
            'coverage_target': self.coverage_target,
            'gamma': gamma,
            'realised_interval_coverage': realised_interval_coverage,
            'alpha_t_list': alpha_t_list,
            'average_prediction_interval': average_prediction_interval,
            'conformal_sets': conformal_sets_list,
            'error_t_list': err_t_list, 
            'interval_size': self.interval_size
        }
        
            
    def DtACI(self, timeseries_data: tuple, gamma_candidates: np.array = None, custom_interval = None):
        xpred, y = timeseries_data
        if gamma_candidates is None:
            gamma_candidates = np.array([0.001, 0.004, 0.032, 0.064, 0.128, 0.256, 0.512])

        # we need a vectorised version of l
        l_vec = self.vectorize_l()
        
        candiate_alpha = np.array([[self.coverage_target for _ in gamma_candidates]])
        gamma_weights = np.array([1 for _ in gamma_candidates])
        
        chosen_gamma_index = []
        err_t_list = []
        conformal_sets_list = []
        alpha_t_list = []
        alpha_error_list = []
        B_t_list = []
        
        sigma = 1/(2*self.interval_size)
        nu = np.sqrt(3/50 * (np.log(len(gamma_candidates)*50) + 2)/((1-self.coverage_target)**2 * self.coverage_target**2))
       
        # Calculating the scores at each time step
        All_scores = self.score_function(xpred, y)

        for i in range(self.interval_size+1, len(All_scores)):
            # Calcualting the probability of each gamma from the weights from step t-1.
            Wt = gamma_weights.sum()
            gamma_probabilites = gamma_weights/Wt
            
            # Choosing a alpha from the probabilites from the gamma candidates.
            chosen_alpha_t = np.random.choice(candiate_alpha[-1], p=gamma_probabilites)
            alpha_t_list.append(chosen_alpha_t)
            candiate_alpha_index = np.where(candiate_alpha[-1] == chosen_alpha_t)[0][0]
            chosen_gamma_index.append(candiate_alpha_index)

            Coverage_t = self.C_t(chosen_alpha_t, All_scores, xpred[i], i, custom_interval)
            conformal_sets_list.append(Coverage_t)
            
            err_true = AdaptiveCP.err_t(y[i], Coverage_t)
            err_t_list.append(err_true)

            # TIME FRONTIER -------

            # Smallest interval containg the true value.
            for possi in np.linspace(1, 0, 1000):
                Cpossi= self.C_t(possi, All_scores, xpred[i], i)
                if Cpossi[0] < y[i] < Cpossi[1]:
                    B_t = possi
                    break
            
            B_t_list.append(B_t)
            
            # Updating the weights.
            new_weights = gamma_weights * np.exp(-nu * l_vec(B_t, candiate_alpha[-1]))
            
            sumW, lenW = sum(new_weights), len(new_weights)
            final_weights = new_weights*(1-sigma) + sumW*(sigma/lenW)
            gamma_weights = final_weights

            # Calculating the coverage and error at each time step, for different alpha values.
            alphai_errors = np.array([AdaptiveCP.err_t(y[i], self.C_t(alpha_i, All_scores, xpred[i], i)) for alpha_i in candiate_alpha[-1]])
            alpha_error_list.append(alphai_errors)

            # Updating the alpha values.
            new_alphas = candiate_alpha[-1] + (gamma_candidates * (self.coverage_target - alphai_errors))
            candiate_alpha = np.vstack((candiate_alpha, new_alphas))

        # Calculating different averages
        realised_interval_coverage = 1 - pd.Series(err_t_list).rolling(self.interval_size).mean().mean()
        average_prediction_interval = np.mean([abs(x[1] - x[0]) for x in conformal_sets_list])

        return {
            'model': 'DtACI',
            'coverage_target': self.coverage_target,
            'gamma_candidates': gamma_candidates,
            'realised_interval_coverage': realised_interval_coverage,
            'average_prediction_interval': average_prediction_interval,
            'alpha_t_list': alpha_t_list,
            'conformal_sets': conformal_sets_list,
            'error_t_list': err_t_list,
            'alpha_error_list': alpha_error_list,
            'B_t_list': B_t_list,
            'interval_size': self.interval_size ,
        }
        

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

        fig, ax = plt.subplots(figsize=figsize)
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

        if 'alpha_t_list' in data:
            alpha_list = data['alpha_t_list']

        if 'B_t_list' in data:
            B_t_list = data['B_t_list']

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


class ACP_data:
    def __init__(self, datapoints: int,  max_seq_length: int = 500, dist_shifts : list[tuple] = None,  random_seq_length: bool = True, trend_seq: float = 0.5, time_series_scale: float = 0.3):
        self.datapoints = datapoints
        self.max_seq_length = max_seq_length
        self.random_seq_length = random_seq_length
        self.trend_seq = trend_seq
        self.time_series_scale = time_series_scale
        
        if dist_shifts is None:
            self.random_dist_shifts = 1
        else:
            self.random_dist_shifts = 0
            self.dist_shifts = dist_shifts


    def _create_timeseries(self, length, given_dist_shifts):
        # Generates the absolute value of difference between timestep
        n = length//len(given_dist_shifts)
        m = length%len(given_dist_shifts)

        final = np.array([])
        
        for i in range(len(given_dist_shifts)-1):
            Y = abs(np.random.normal(given_dist_shifts[i][0], given_dist_shifts[i][1], n))
            final = np.concatenate((final, Y))
        
        final = np.concatenate((final, abs(np.random.normal(given_dist_shifts[-1][0], given_dist_shifts[-1][1], n+m)))) 
        
        return final + self.time_series_scale*np.roll(final, 1)

    def generate(self) -> list[tuple]:
        generated_data = []
        max_mean, max_var = 10, 10  # Later will make this user defined.

        # This control the upward or negatice skew of the data.
        pdist = [1- self.trend_seq, self.trend_seq]
 
        for _ in range(self.datapoints):
            if self.random_seq_length:             
                # Up to user to make seq_lenght // 3 greater then lookback window.
                length = randint(self.max_seq_length//3, self.max_seq_length)
            else:
                length = self.max_seq_length
            
            # Direciton of the time series
            X =  np.random.choice([-1, 1], size=length, p=pdist)
            
            if self.random_dist_shifts:
                # Randomly generating the distribution shifts.
                dist_shift = [(randint(-max_mean, max_mean), randint(0, max_var)) for __ in range(randint(1, randint(1, length//50)))]
            else:
                dist_shift = self.dist_shifts
            
            Y = self._create_timeseries(length, dist_shift)
            T = np.cumsum(X*Y)

            input_data, labels_data = T[:-1], T[1:]
            generated_data.append((input_data, labels_data))

        return generated_data




if __name__ == '__main__':
    # This creates a TS, where the distribution changes at different points deppending on a provided list of distribution shifts.
    def Create_TimeSeries(s, dist_shifts):
        n = s//len(dist_shifts)
        m = s%len(dist_shifts)

        final = np.array([])
        
        for i in range(len(dist_shifts)-1):
            Y = abs(np.random.normal(dist_shifts[i][0], dist_shifts[i][1], n))
            final = np.concatenate((final, Y))
        
        final = np.concatenate((final, abs(np.random.normal(dist_shifts[-1][0], dist_shifts[-1][1], n+m))))

        return final

    # Here we will just look at a simple random walk.

    p = 0.5
    pdist = [1-p, p]
    steps = [-1, 1]

    datapoints = 10
    seq_length = 300

    output_label_tuples = []

    dist_shift = [(-10, 1)]

    for _ in range(datapoints):
        X = np.random.choice(steps, size=seq_length, p=pdist)
        Y = Create_TimeSeries(seq_length, dist_shift)
        T = np.cumsum(X*Y)
        
        input_data = T[:-1]
        labels_data = T[1:]

        output_label_tuples.append((input_data, labels_data))
        

    ACP = AdaptiveCP(0.3, 50)

    for i, data in enumerate(output_label_tuples):
        a = ACP.ACI(data, 0.05)
        b = ACP.NonAdaptive(data)
        c = ACP.DtACI(data)

        ACP_plots.plot_alpha_t([a, b, c])
        break