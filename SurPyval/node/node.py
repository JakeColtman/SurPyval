from scipy.stats import rv_continuous
import scipy.stats
from typing import Dict, Any


class Node:
    """
        A node in the model's graphical model

        Generally anything that contributes to the joint probability will be a node
        (see by contrast Transformations that don't contribute to the pdf)

        The primary job of the Node class is to handle routing between SurPyval concepts and scipy classes

        Parameters
        ----------
        distribution: rv_continuous
                      a scipy style distribution (https://docs.scipy.org/doc/scipy/reference/stats.html)
        parameter_dict: Dict[str, str]
                        a mapping from Parameter name to scipy arg name, e.g. {"alpha": "shape", "beta": "scale"}
        constants_dict: Dict[str, array-like] optional
                        a mapping to hard code arguments to, e.g., set hyper parameters

        Examples
        --------
        For nodes that are completely determined by parameters of the model, the parameter dict should contain
        all relevant pieces of information.  Here we create a gamma node over beta with shape and scale set by parameters
        alpha and llambda

        >>> from scipy.stats import gamma
        >>> parameter_dict = {"alpha": "shape", "llambda": "scale", "beta": "x"}
        >>> node = Node(gamma, parameter_dict)

        Often we will have fixed values to use for certain arguments (e.g. for setting hyper parameters).
        In this case we can set them through the optional constants dicts

        >>> from scipy.stats import gamma
        >>> parameter_dict = {"beta": "x"}
        >>> node = Node(gamma, parameter_dict, {"scale": 1.0, "loc": 10.0})
    """

    def __init__(self, distribution: rv_continuous, parameter_dict: Dict[str, str], constants_dict: Dict[str,Any]=None):
        self.distribution = distribution
        self.parameter_dict = parameter_dict
        self.constants_dict = constants_dict if constants_dict is not None else {}
        self.parameter_names = parameter_dict.keys()

    def parse_unflattened_parameters(self, **kwargs):
        filtered_kwargs = {x: kwargs[x] for x in kwargs if x in self.parameter_names}
        renamed_kwargs = {self.parameter_dict[x]: filtered_kwargs[x] for x in filtered_kwargs}
        return {**renamed_kwargs, **self.constants_dict}

    def logpdf(self, **kwargs):
        return self.distribution.logpdf(**self.parse_unflattened_parameters(**kwargs))

    def pdf(self, **kwargs):
        return self.distribution.logpdf(**self.parse_unflattened_parameters(**kwargs))

    def ppf(self, **kwargs):
        return self.distribution.ppf(**self.parse_unflattened_parameters(**kwargs))


def gaussian(parameter_dict, constant_dict=None):
    distr = scipy.stats.norm
    return Node(distr, parameter_dict, constant_dict)


def exponential(parameter_dict, constant_dict=None):
    distr = scipy.stats.expon
    return Node(distr, parameter_dict, constant_dict)


def gamma(parameter_dict: Dict[str, str], constant_dict=None):
    distr = scipy.stats.gamma
    return Node(distr, parameter_dict, constant_dict)
