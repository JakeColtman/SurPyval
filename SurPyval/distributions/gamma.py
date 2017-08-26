from numpy.random import gamma
from SurPyval.core.sampling import NumpySampler

class Gamma(NumpySampler):
    
    def __init__(self, alpha, llambda):
        self.sampler = NumpySampler(gamma, shape = alpha, scale = 1./ llambda)

    def sample(self, n_samples):
        return self.sampler.sample(n_samples)

def gamma_from_mean_variance(mean, variance):
    llambda = mean / variance
    alpha = llambda * mean
    return Gamma(alpha, llambda)