from __future__ import division
import numpy as np
from numpy.random import gamma
from scipy.optimize import minimize
from SurPyval.distributions.gamma import Gamma   

class ConjugatePriorFittedExponential:

    def __init__(self, data, event):
        self.data = data
        self.event = event
        self.fit()

    @staticmethod
    def log_lik(data, event, l):
        d = np.sum(event)
        y_sum = np.sum(data)
        return d * np.log(l) - l * y_sum

    def fit(self, gamma_prior):
        d = np.sum(self.event)
        y_sum = np.sum(self.data)
        self.fitted_l = Gamma(gamma_prior.alpha + d, gamma_prior.llambda + y_sum)
        #self.fitted_map_log_lik = self.log_lik(self.data, self.event, self.fitted_l.sample(1000).mean())
        return self

    def sample_posterior_predictive(self, n_samples):
        return np.random.exponential(self.fitted_l, n_samples)

    def sample_survival_function(self, y, n_samples):
        l_s = self.fitted_l.sample(n_samples)
        return np.array(map(lambda l: np.exp(-l * y) , l_s))
    