from SurPyval.distributions.exponential import Exponential
from SurPyval.core.sampling import NumpySampler
from scipy.optimize import minimize
import numpy as np

class MaximumLikihoodFittedExponential:

    def __init__(self, y_s, event):
        self.y_s = y_s
        self.event = event

    @staticmethod
    def log_lik(data, event, l):
        d = np.sum(event)
        y_sum = np.sum(data)
        return d * np.log(l) - l * y_sum

    def fit(self, starting_point = None):
        def function_to_minimize(beta):
            return -1 * self.log_lik(self.y_s, self.event, beta)
        if starting_point is None:
            starting_point = 0.5
        result = minimize(function_to_minimize, starting_point)
        self.fitted_l = result["x"]
        self.log_lik = self.log_lik(self.y_s, self.event, self.fitted_l)
        return self

    def posterior_predictive(self):
        return NumpySampler(np.random.exponential, scale = self.fitted_l)

    def survival_function(self, y):
        l_s = self.fitted_l * y
        return np.exp(l_s * -1)