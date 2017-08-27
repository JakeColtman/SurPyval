from SurPyval.distributions.exponential import Exponential
from SurPyval.core.sampling import NumpySampler
from scipy.optimize import minimize 
import numpy as np

class MaximumLikihoodFittedExponentialRegression:

    def __init__(self, y_s, x_s, event):
        self.y_s = y_s
        self.x_s = x_s
        self.event = event

    @staticmethod
    def log_lik(y_s, x_s, event, beta):
        llambda = np.dot(x_s, beta)
        return np.dot(event , llambda) - np.dot(y_s , np.exp(llambda))

    def fit(self, starting_point = None):
        def function_to_minimize(beta):
            return -1 * self.log_lik(self.y_s, self.x_s, self.event, beta)
        if starting_point is None:
            starting_point = tuple([0.5] * self.x_s.shape[1])
        result = minimize(function_to_minimize, starting_point)
        self.fitted_l = result["x"] * -1
        self.log_lik = self.log_lik(self.y_s, self.x_s, self.event, self.fitted_l)
        return self

    def posterior_predictive(self, x):
        llambda = np.exp(np.dot(x, self.fitted_l))
        return NumpySampler(np.random.exponential, scale = llambda)

    def survival_function(self, y, x):
        l_s = np.exp(np.dot(x, self.fitted_l)) * y
        return np.exp(l_s * -1)


