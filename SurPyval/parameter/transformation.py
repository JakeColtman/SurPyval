from typing import Callable, Dict, Any

class Transformation(object):
    """
        Allows the definition of composite "parameters" to add to the parameter dictionary

        Useful for simplifying the code of other nodes by pre-baking calculations
        e.g. in a regression model, the linear predictor could be a transformation.

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

    def transform(self, data_dict, parameter_dict):
        """
        Apply the transformation function to data and parameters

        Returns
        -------
        array-like
        """
        return {self.name: self.f_transformation(data_dict, parameter_dict)}