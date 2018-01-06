from numpy.random import gamma
import seaborn as sns
import numpy as np

from matplotlib import pyplot as plt

from SurPyval.core.sampling import NumpySampler
from SurPyval.distributions.distribution import Distribution

class Gamma(Distribution):
    
    def __init__(self, alpha, llambda):
        self.alpha, self.llambda = alpha, llambda
        self.sampler = NumpySampler(gamma, shape = alpha, scale = 1./ llambda)

    def pdf(self, x):
        from scipy.stats import gamma
        return gamma.pdf(x, shape = self.alpha, scale = 1./self.llambda)

    def log_lik(self, x):
        return np.log(self.pdf(x))