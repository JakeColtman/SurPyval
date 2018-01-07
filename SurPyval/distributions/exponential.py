from numpy.random import exponential
import numpy as np

from SurPyval.samplers.npsampler import NumpySampler
from SurPyval.distributions.distribution import Distribution

class Exponential(Distribution):
    
    def __init__(self, alpha):
        self.alpha = alpha
        self.sampler = NumpySampler(exponential, shape = self.alpha)

    def pdf(self, x):
        from scipy.stats import exponential
        if x <= 0:
            return - np.inf
        return exponential.pdf(x, shape = self.alpha)

    def log_lik(self, x):
        return np.log(self.pdf(x))