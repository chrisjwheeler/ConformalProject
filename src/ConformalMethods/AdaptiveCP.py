import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Callable
from random import randint


class AdaptiveCP:
    def __init__(self, coverage_target=0.1, interval_size: int =50, score_function=None, neg_inverse_score=None, pos_inverse_score=None):
        self.coverage_target = coverage_target
        self.interval_size = interval_size
        self.score_function = score_function if score_function is not None else lambda xpred, y: (abs(y - xpred))/abs(xpred)
        self.neg_inverse_score = neg_inverse_score if neg_inverse_score is not None else lambda x_t, Q: (x_t) - (abs(x_t) * Q)
        self.pos_inverse_score = pos_inverse_score if pos_inverse_score is not None else lambda x_t, Q: (x_t) + (abs(x_t) * Q)

    def C_t(self, alpha_t, scores, x_t, t, custom_interval = None):
        interval = custom_interval if custom_interval is not None else self.interval_size
        assert interval < len(scores), 'Attempting to look back further than is possible.'
        
        # Insuring that alpha_t is between 0 and 1
        alpha_t = min(1, max(0, alpha_t))
        
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

    def ACI(self, timeseries_data: tuple, gamma: float, custom_interval = None, title: str = None) -> dict:
        xpred, y = timeseries_data
        alpha_t_list = [self.coverage_target]

        All_scores = self.score_function(xpred, y)

        err_t_list = []
        conformal_sets_list = []

        if custom_interval is not None:
            bigger_interval = max(custom_interval, self.interval_size) + 1
        else:
            bigger_interval = self.interval_size + 1
        
        for i in range(bigger_interval, len(All_scores)):
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
            'model': title if title is not None else 'ACI',
            'coverage_target': self.coverage_target,
            'gamma': gamma,
            'realised_interval_coverage': realised_interval_coverage,
            'alpha_t_list': alpha_t_list,
            'average_prediction_interval': average_prediction_interval,
            'conformal_sets': conformal_sets_list,
            'error_t_list': err_t_list, 
            'interval_size': self.interval_size
        }
        
            
    def DtACI(self, timeseries_data: tuple, gamma_candidates: np.array = None, custom_interval: int = None, title: str = None) -> dict:
        xpred, y = timeseries_data
        if gamma_candidates is None:
            gamma_candidates = np.array([0.001, 0.004, 0.032, 0.064, 0.128, 0.256, 0.512])

        if custom_interval is not None:
            bigger_interval = max(custom_interval, self.interval_size) + 1
        else:
            bigger_interval = self.interval_size + 1

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
        nu = np.sqrt((3/self.interval_size) * (np.log(len(gamma_candidates)*self.interval_size) + 2)/((1-self.coverage_target)**2 * self.coverage_target**2))
       
        # Calculating the scores at each time step
        All_scores = self.score_function(xpred, y)

        for i in range(bigger_interval, len(All_scores)):
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
            B_t = 0.5       # To avoid unbound local error will assign B_t a value first
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
            'model': title if title is not None else 'DtACI',
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