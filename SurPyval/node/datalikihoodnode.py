import numpy as np
from typing import Dict, Any

from SurPyval.node.node import Node


class DataLikihoodNode(Node):
    """
    Extension of a standard Node to automatically take account of censoring

    Uncensored observations contribute their pdf to jointlikihood
    Censored events contribute only their survival function

    DataLikihoodNode automatically does this routing
    """

    def __init__(self, distribution, data: Any, parameter_dict: Dict[str, str], constants_dict: Dict[str, Any]=None):
        self.data = data
        Node.__init__(self, distribution, parameter_dict, constants_dict)

    def logpdf(self, **kwargs):
        processed_kwargs = self.parse_unflattened_parameters(**kwargs)
        likihood_contribution = self.distribution.logpdf(kwargs["y"], **processed_kwargs)
        survival_contribution = self.distribution.logsf(kwargs["y"], **processed_kwargs)
        return np.sum(likihood_contribution[kwargs["event"].astype(bool)]) + np.sum(survival_contribution[~kwargs["event"].astype(bool)])