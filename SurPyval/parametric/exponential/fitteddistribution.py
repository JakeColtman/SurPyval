import numpy as np
from scipy.optimize import minimize 

from functools import partial

from collections import namedtuple

Parameterization = namedtuple("Parameterization", ["log_lik_function", "num_parameters", "bounds"])

def exponential_log_lik(y, e, l):
    d = np.sum(e)
    return -(np.sum(y) * l) + np.log(l ** d)

def weibull_log_lik(y, e, a, l):    
    event = np.log(a) + (a - 1) * np.log(y) + l - (np.exp(l) * (y ** a))
    censored = - (np.exp(l) * (y ** a))
    return np.sum(e * event + (1 - e) * censored)

ExponentialParameterization = Parameterization(
                                exponential_log_lik,
                                1,
                                ((None, None)))

WeibullParameterization =     Parameterization(
                                weibull_log_lik,
                                2,
                                ((0, None), (None, None)))


class FittedDistribution(object):
    def __init__(self, parameterization, lifetimes, events):
        self.lifetimes, self.events, self.parameterization = lifetimes, events, parameterization
        self.num_events = np.sum(self.events)
        self._fit()

    def _fit(self, method = "Nelder-Mead"):
        funct_to_min = partial(self.parameterization.log_lik_function, {"y": self.lifetimes, "e": self.events})
        self.optimal_params = minimize(lambda x: funct_to_min(x), bounds = self.parameterization.bounds)

    def log_lik(self):
        """ Log likihood of the most likely fitted parameters
        """
        return

    def
    



