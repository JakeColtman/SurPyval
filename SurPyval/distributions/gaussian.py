from numpy.random import multivariate_normal
import numpy as np

from SurPyval.samplers.npsampler import NumpySampler
from SurPyval.distributions.distribution import Distribution

class Gaussian(Distribution):
    
    def __init__(self, mu, covar):
        if type(covar) is float:
            covar = np.array([[covar]])
        if type(mu) is float:
            mu = np.array([mu])
        self.mu, self.covar = mu, covar

        self.sampler = NumpySampler(multivariate_normal, 
                                    mean = self.mu, 
                                    cov = self.covar)

    def pdf(self, x):
        from scipy.stats import multivariate_normal
        return multivariate_normal.pdf(x, mean = self.mu, cov = self.covar)

    def log_lik(self, x):
        return np.log(self.pdf(x))