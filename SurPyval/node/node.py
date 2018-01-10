from SurPyval.distributions.gaussian import Gaussian
from SurPyval.distributions.exponential import Exponential
from SurPyval.distributions.gamma import Gamma


class Node:
    """
        A node in the model's graphical model / node tree

        Each node represents a variable in the model

        Nodes act as an anticorruption layer between NodeTree and the internal workings of samplers and distributions
        They:
            * Provide a common API
            * Route model parameters to the correct distribution parameters
    """

    def __init__(self, distribution, sampler, parameter_dict):
        self.distribution = distribution
        self.sampler = sampler
        self.parameter_dict = parameter_dict
        self.parameter_names = parameter_dict.keys()

    def sample(self, n_samples):
        return self.sampler.sample(n_samples)

    def log_lik(self, **kwargs):
        filtered_kwargs = {x: kwargs[x] for x in kwargs if x in self.parameter_names}
        renamed_kwargs = {self.parameter_dict[x]: filtered_kwargs[x] for x in filtered_kwargs}
        return self.distribution.log_lik(**renamed_kwargs)
    
    def pdf(self, **kwargs):
        kwargs = {x: kwargs[x] for x in kwargs if x in self.parameter_names}
        return self.distribution.pdf(**kwargs)


def gaussian(mu, covar, parameter_dict):
    distr = Gaussian(mu, covar)
    sampler = distr.sampler
    return Node(distr, sampler, parameter_dict)


def exponential(alpha, parameter_dict):
    distr = Exponential(alpha)
    sampler = distr.sampler
    return Node(distr, sampler, parameter_dict)


def gamma_from_mean_variance(mean, variance, parameter_dict):
    llambda = mean / variance
    alpha = llambda * mean
    distr = Gamma(alpha, llambda)
    sampler = distr.sampler
    return Node(distr, sampler, parameter_dict)
