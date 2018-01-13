from typing import Callable, Dict, Any

from SurPyval.node.node import Node


class DeterministicNode(Node):
    """
        Define deterministic transformations of other variables in the model

        Useful for simplifying the code of other nodes by pre-baking calculations
        e.g. in a regression model, the linear predictor could be a transformation.

        DeterministicNodes do not contribute to the loglikihood directly

        Parameters
        ----------
        f_transformation: NodeTree
                          Function to apply to parameters and data_dict to produce new parameter
                          Must take as parameters (data_dict, parameter_dict) and return an array-like
        name: str
                  Name of the resulting parameter as should appear in parameter_dict
    """

    def __init__(self, f_transformation: Callable[[Dict[str, Any], Dict[str, Any]], Any], name):
        self.f_transformation = f_transformation
        self.name = name

    def logpdf(self, **kwargs) -> float:
        return 0.0

    def pdf(self, **kwargs) -> float:
        return 1.0

    def ppf(self, **kwargs):
        raise NotImplementedError("ppf not defined for DeterministicNode")

    def transform(self, data_dict, parameter_dict):
        """
        Apply the transformation function to data and parameters

        Returns
        -------
        array-like
        """
        return {self.name: self.f_transformation(data_dict, parameter_dict)}