import numpy as np

from SurPyval.node.node import Node


class DataLikihoodNode(Node):

    def logpdf(self, **kwargs):
        processed_kwargs = self.parse_unflattened_parameters(**kwargs)
        likihood_contribution = self.distribution.logpdf(kwargs["y"], **processed_kwargs)
        survival_contribution = self.distribution.logsf(kwargs["y"], **processed_kwargs)
        return np.sum(likihood_contribution[kwargs["event"].astype(bool)]) + np.sum(survival_contribution[~kwargs["event"].astype(bool)])