import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Callable
import random
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

        ## Trial code

        if 1 - alpha_t == 0:
            Q = 0
        elif 1 - alpha_t == 1:  
            Q = max(scores[t-interval:t])
        else:      
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

    def ACI(self, timeseries_data: tuple, gamma: float = 0.05, custom_interval = None, title: str = None, startpoint: int = None) -> dict:
        xpred, y = timeseries_data
        alpha_t_list = [self.coverage_target]

        All_scores = self.score_function(xpred, y)

        err_t_list = []
        conformal_sets_list = []

        if startpoint is None:
            if custom_interval is not None:
                startpoint = max(custom_interval, self.interval_size) + 1
            else:
                startpoint = self.interval_size + 1
        
        for i in range(startpoint, len(All_scores)):
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
        
            
    def DtACI(self, timeseries_data: tuple, gamma_candidates: np.array = None, custom_interval: int = None, title: str = None, startpoint: int = None) -> dict:
        xpred, y = timeseries_data
        if gamma_candidates is None:
            gamma_candidates = np.array([0.001, 0.004, 0.032, 0.064, 0.128, 0.256, 0.512])

        if startpoint is None:
            if custom_interval is not None:
                startpoint = max(custom_interval, self.interval_size) + 1
            else:
                startpoint = self.interval_size + 1

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

        for i in range(startpoint, len(All_scores)):
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

            # Smallest interval containg the true value. Found using binary search.
            low, high = 0, 999 
            possibilities = np.linspace(0, 1, 1000) # as 1 - 

            B_t = 1
            while low <= high:
                mid = (high + low) // 2
                possi = possibilities[mid]
                Cpossi = self.C_t(possi, All_scores, xpred[i], i, custom_interval)

                if Cpossi[0] < y[i] < Cpossi[1]:
                    B_t = possi
                    low = mid + 1
                else:
                    high = mid - 1
            
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
    
    def set_loss(self, optimal_set, given_set):
        # If the optimal set is somehow 0, then we will return the given set.
        if optimal_set == 0:
            return 0
        else:
            val = (optimal_set - given_set) / optimal_set
         
        if val < 0:
            return (self.coverage_target) * (-1* val)
        else:
            return (1 - self.coverage_target) * val
    
    def set_loss_vectorize(self):
         return np.vectorize(self.set_loss)
    
    def ACI_head(self, timeseries_data: tuple, gamma: float, start_point: int, custom_interval = None):
        """
        Creates an ACI head that calculates the coverage set for each data point in the time series.

        Args:
            timeseries_data (tuple): A tuple containing the time series data. The first element is the predicted values (xpred),
                                    and the second element is the actual values (y).
            gamma (float): The step size for updating the coverage target.
            start_point (int): The index of the starting point in the time series data.
            custom_interval (optional): A custom interval to use for calculating the coverage. Defaults to None.

        Yields:
            list: The coverage set for each data point in the time series.

        Returns:
            bool: False indicating the completion of the method.
        """
        xpred, y = timeseries_data
        alpha_t_list = [self.coverage_target]

        All_scores = self.score_function(xpred, y)

        err_t_list = []
        conformal_sets_list = []
        
        for i in range(start_point, len(All_scores)):
            Coverage_t = self.C_t(alpha_t_list[-1], All_scores, xpred[i], i, custom_interval)
            conformal_sets_list.append(Coverage_t)

            yield Coverage_t

            error_t = AdaptiveCP.err_t(y[i], Coverage_t)
            err_t_list.append(error_t)

            alpha_t = min(max(alpha_t_list[-1] + (gamma * (self.coverage_target - error_t)), 0), 1)
            alpha_t_list.append(alpha_t)

        return False
    
    def AwACI(self, timeseries_data: tuple, interval_candidates: np.array = None, nu_sigma: tuple = (10**-3, 0.05), gamma: float = 0.05, title: str = None):
        
        xpred, y = timeseries_data

        chosen_interval_index = []
        err_t_list = []
        conformal_sets_list = []
        optimal_radius_list = []
        chosen_radius_list = []

        Set_loss = self.set_loss_vectorize()

        # Scale parameters, havent looked into scaling them best.
        sigma = nu_sigma[1]
        nu = nu_sigma[0] 

        if interval_candidates is None:
            interval_candidates = np.array(range(50, 550, 100))

        # To sync all of the heads we need to start at the max of all the candidates.
        start_point = max(interval_candidates) + 1
        i_count = start_point

        # Create the head and intitialse the weights.
        ACI_heads = [self.ACI_head(timeseries_data, gamma, start_point, interval) for interval in interval_candidates]
        interval_weights = np.array([1 for _ in range(len(interval_candidates))])
        
        # Continues calculating intervals until one of the heads stops.
        none_terminated = True

        while none_terminated: 
            head_sets = [] # Will contain the result from each head.
            
            # Create the mass distribution for each head
            Wt = interval_weights.sum()

            interval_probabilites = interval_weights/Wt
        
            try:
                # Create a list of the coverages for the different heads.
                for head in ACI_heads:
                        head_sets.append(next(head))
            
            except StopIteration: # One head is terminated.
                none_terminated = False
                break # You could but the return statement here

            # Choosing which head to use.
            chosen_set = random.choices(head_sets, weights=interval_probabilites, k=1)[0] # Using random module as numpy can not deal with tuples.
            conformal_sets_list.append(chosen_set)
            chosen_interval_index.append(head_sets.index(chosen_set))

            # TIME FRONTIER -------------

            # Seeing whether result lies within the set.
            err_true = AdaptiveCP.err_t(y[i_count], chosen_set)
            err_t_list.append(err_true)

            # Computing the conformal set radi. 
            optimal_set_radius = abs(y[i_count] - y[i_count-1]) 
            head_set_radius = list(map(lambda Cset: (Cset[1] - Cset[0])/2, head_sets)) #(chosen_set[1] - chosen_set[0])/2

            optimal_radius_list.append(optimal_set_radius)
            chosen_radius_list.append((chosen_set[1] - chosen_set[0])/2)
            
            head_set_radius = np.array(head_set_radius)

            # Updating the weights.
            new_weights = interval_weights * np.exp(-1 * nu * Set_loss(optimal_set_radius, head_set_radius)) 
            sumW, lenW = sum(new_weights), len(new_weights)
            final_weights = new_weights*(1-sigma) + sumW*(sigma/lenW)
            interval_weights = final_weights

            # Incrementing the i-count
            i_count+=1

        # Calculating different averages
        realised_interval_coverage = 1 - pd.Series(err_t_list).rolling(50).mean().mean() # 50 is arbitary and could be improved.
        average_prediction_interval = np.mean([abs(x[1] - x[0]) for x in conformal_sets_list])

        return {
                'model': title if title is not None else 'AwACI',
                'coverage_target': self.coverage_target,
                'interval_candidates': interval_candidates,
                'realised_interval_coverage': realised_interval_coverage,
                'average_prediction_interval': average_prediction_interval,
                'optimal_set_radius': optimal_radius_list, 
                'chosen_set_radius': chosen_radius_list,
                'conformal_sets': conformal_sets_list,
                'error_t_list': err_t_list,
                'chosen_interval_index': chosen_interval_index,
                'start_point': start_point,
                'interval_size': 50
            }
    
    def DtACI_head(self, timeseries_data: tuple, custom_interval: int = None, start_point: int = None, gamma_candidates: np.array = None,  title: str = None):
        '''start_point: The value which the head will start calculating from'''
        
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
        
        sigma = 1/(2*custom_interval)
        nu = np.sqrt((3/custom_interval) * (np.log(len(gamma_candidates)*custom_interval) + 2)/((1-self.coverage_target)**2 * self.coverage_target**2))

        # Calculating the scores at each time step
        All_scores = self.score_function(xpred, y)

        for i in range(start_point, len(All_scores)):
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

            # Yield the conformal set
            yield Coverage_t
            
            err_true = AdaptiveCP.err_t(y[i], Coverage_t)
            err_t_list.append(err_true)

            # TIME FRONTIER -------

            # Smallest interval containg the true value.
            low, high = 0, 999 
            possibilities = np.linspace(0, 1, 1000) # as 1 - 

            B_t = 1
            while low <= high:
                mid = (high + low) // 2
                possi = possibilities[mid]
                Cpossi = self.C_t(possi, All_scores, xpred[i], i, custom_interval)

                if Cpossi[0] < y[i] < Cpossi[1]:
                    B_t = possi
                    low = mid + 1
                else:
                    high = mid - 1
            
            B_t_list.append(B_t)
            
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
        
        return False

    def AwDtACI(self, timeseries_data: tuple, interval_candidates: np.array = None, nu_sigma: tuple = (10**-3, 0.05), title: str = None):
        '''nu_sigma: The first value is the nu value and the second is the sigma value.'''
        xpred, y = timeseries_data

        chosen_interval_index = []
        err_t_list = []
        conformal_sets_list = []
        optimal_radius_list = []
        chosen_radius_list = []

        if interval_candidates is None:
            interval_candidates = np.array(range(50, 550, 100))

        # To sync all of the heads we need to start at the max of all the candidates.
        start_point = max(interval_candidates) + 1

        # Create the head and intitialse the weights.
        DtACI_heads = [self.DtACI_head(timeseries_data, interval, start_point) for interval in interval_candidates]
        interval_weights = np.array([1 for _ in range(len(interval_candidates))])

        # Scaling the parameters, there is no way these are the best ways.
        sigma = nu_sigma[1]
        nu = nu_sigma[0] 
        
        l_vec = self.set_loss_vectorize()

        none_terminated = True
        i_count = start_point
        
        while none_terminated: # Continues calculating intervals until one of the heads stops.
            head_sets = []
            
            # Create the mass distribution for each head
            Wt = interval_weights.sum()
            interval_probabilites = interval_weights/Wt
        
            # Create a list of the coverages for the different heads.
            try:
                for head in DtACI_heads:
                        head_sets.append(next(head))
            
            except StopIteration: # The head is terminated.
                none_terminated = False
                break # You could but the return statement here

            # Choosing which head to use.
            chosen_set = random.choices(head_sets, weights=interval_probabilites, k=1)[0] # Using random module as numpy can not deal with tuples.
            conformal_sets_list.append(chosen_set)
            chosen_interval_index.append(head_sets.index(chosen_set))

            # TIME FRONTIER -------------

            # Seeing whether result lies within the set.
            err_true = AdaptiveCP.err_t(y[i_count], chosen_set)
            err_t_list.append(err_true)

            # Finding the best possible set. 
            optimal_set_radius = xpred[i_count] - xpred[i_count-1]
            head_set_radius = list(map(lambda Cset: (Cset[1] - Cset[0])/2, head_sets)) #(chosen_set[1] - chosen_set[0])/2

            optimal_radius_list.append(optimal_set_radius)
            chosen_radius_list.append((chosen_set[1] - chosen_set[0]/2))
            
            #l1_error = abs(optimal_set_radius - np.array(head_set_radius))
            head_set_radius = np.array(head_set_radius)
            l1_error = l_vec(optimal_set_radius, head_set_radius)

            # Updating the weights.
            new_weights = interval_weights * np.exp(-1 * nu * l1_error) # Removed negative from previous paper.
            
            sumW, lenW = sum(new_weights), len(new_weights)
            final_weights = new_weights*(1-sigma) + sumW*(sigma/lenW)
            interval_weights = final_weights

            # Incrementing the i-count
            i_count+=1

        # Calculating different averages
        realised_interval_coverage = 1 - pd.Series(err_t_list).rolling(50).mean().mean() # 50 is arbitary and could be improved.
        average_prediction_interval = np.mean([abs(x[1] - x[0]) for x in conformal_sets_list])

        return {
                'model': title if title is not None else 'AwDtACI',
                'coverage_target': self.coverage_target,
                'interval_candidates': interval_candidates,
                'realised_interval_coverage': realised_interval_coverage,
                'average_prediction_interval': average_prediction_interval,
                'optimal_set_radius': optimal_radius_list, 
                'chosen_set_radius': chosen_radius_list,
                'conformal_sets': conformal_sets_list,
                'error_t_list': err_t_list,
                'chosen_interval_index': chosen_interval_index,
                'start_point': start_point,
                'interval_size': 50
            }

