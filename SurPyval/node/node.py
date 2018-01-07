from SurPyval.distributions.gaussian import Gaussian
from SurPyval.distributions.exponential import Exponential
from SurPyval.distributions.gamma import Gamma

class Node:

    def __init__(self, distribution, sampler, parameter_names):
        self.distribution = distribution
        self.sampler = sampler
        self.parameter_names = parameter_names

    def sample(self, n_samples):
        return self.sampler.sample(n_samples)

    def log_lik(self, **kwargs):
        args = [kwargs[x] for x in self.parameter_names]
        return self.distribution.log_lik(*args)
    
    def pdf(self, **kwargs):
        args = [kwargs[x] for x in self.parameter_names]
        return self.distribution.pdf(*args)

def gaussian(mu, covar, variable_names):
    distr = Gaussian(mu, covar)
    sampler = distr.sampler
    return Node(distr, sampler, variable_names)

def exponential(alpha, variable_names):
    distr = Exponential(alpha)
    sampler = distr.sampler
    return Node(distr, sampler, variable_names)
    
def gamma_from_mean_variance(mean, variance, variable_names):
    llambda = mean / variance
    alpha = llambda * mean
    distr = Gamma(alpha, llambda)
    sampler = distr.sampler
    return Node(distr, sampler, variable_names)