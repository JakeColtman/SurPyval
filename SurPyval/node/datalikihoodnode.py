 import numpy as np

 from SurPyval.node.node import Node

 class DataLikihoodNode(Node):

    def logpdf(self, **kwargs):
        processed_kwargs = self.parse_unflattened_parameters(**kwargs)
        likihood_contribution = self.distribution.logpdf(kwargs["y"][kwargs["event"].astype(bool)], **processed_kwargs)
        survival_contribution = self.distribution.logsf(kwargs["y"][~kwargs["event"].astype(bool)], **processed_kwargs)
        return np.sum(likihood_contribution) + np.sum(survival_contribution)