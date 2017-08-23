from __future__ import division
import numpy as np
from numpy.random import gamma
from scipy.optimize import minimize
from SurPyval.distributions.gamma import Gamma   

class NonConjugatePriorFittedExponential:

    def __init__(self, data, event):
        self.data = data
        self.event = event
        self.fit()

    @staticmethod
    def log_lik(data, event, l):
        d = np.sum(event)
        y_sum = np.sum(data)
        return d * np.log(l) - l * y_sum
        
    def fit(self, starting_point = 0.1):
        def function_to_minimize(parameter):
            return -1 * self.log_lik(self.data, self.event, parameter)
        result = minimize(function_to_minimize, starting_point, bounds = ((0, None), ))
        self.fitted_l = result["x"][0]
        self.fitted_map_log_lik = self.log_lik(self.data, self.event, self.fitted_l)
        return self
    
    def sample_posterior_predictive(self, n_samples):
        l_s = self.fitted_l.sample(n_samples)
        return np.array(map(lambda l: np.random.exponential(1 / l), l_s))
    
    def sample_survival_function(self, y, n_samples):
        l_s = self.fitted_l.sample(n_samples)
        return np.array(map(lambda l: np.exp(-l * y) , l_s))
    