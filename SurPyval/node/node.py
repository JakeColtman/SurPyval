from scipy.stats import rv_continuous
import scipy.stats
from typing import Dict


class Node:
    """
        A node in the model's graphical model / node tree

        Each node represents a variable in the model

        They:
            * Provide a common API
            * Route model parameters to the correct distribution parameters
    """

    def __init__(self, distribution: rv_continuous, parameter_dict: Dict[str, str]):
        self.distribution = distribution
        self.parameter_dict = parameter_dict
        self.parameter_names = parameter_dict.keys()

    def parse_unflattened_parameters(self, **kwargs):
        filtered_kwargs = {x: kwargs[x] for x in kwargs if x in self.parameter_names}
        return {self.parameter_dict[x]: filtered_kwargs[x] for x in filtered_kwargs}

    def logpdf(self, **kwargs):
        return self.distribution.logpdf(**self.parse_unflattened_parameters(**kwargs))

    def pdf(self, **kwargs):
        return self.distribution.logpdf(**self.parse_unflattened_parameters(**kwargs))

    def ppf(self, **kwargs):
        return self.distribution.ppf(**self.parse_unflattened_parameters(**kwargs))


def gaussian(parameter_dict):
    distr = scipy.stats.norm
    return Node(distr, parameter_dict)


def exponential(parameter_dict):
    distr = scipy.stats.expon
    return Node(distr, parameter_dict)


def gamma_from_mean_variance(parameter_dict: Dict[str, str]):
    distr = scipy.stats.gamma
    return Node(distr, parameter_dict)
