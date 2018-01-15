from typing import Dict, Any
import numpy as np

from SurPyval.node import NodeTree, Node
from SurPyval.samplers import EmceeSampler


class FitModel:
    """
        A model that has a posterior sampler

        Parameters
        ----------
        node_tree: NodeTree
                   The graphical structure of the model
        posterior: EmceeSampler
                   sampler for draws of the parameters from the posterior distribution

    """

    def __init__(self, node_tree: NodeTree, posterior: EmceeSampler):
        self.node_tree = node_tree
        self.posterior = posterior

    def sample_replicate(self):
        posterior_sample_parameters = self.posterior.sample(1)[0]
        return self.node_tree.generate_replicate(posterior_sample_parameters)

    def generate_replicates(self, n_replicates: int):
        posterior_samples = self.posterior.sample(n_replicates)[:n_replicates]
        return [self.node_tree.generate_replicate(posterior_sample) for posterior_sample in posterior_samples]

    def predict(self, node_dict: Dict[str, Node]):
        fitted_node_tree = self.node_tree.update(node_dict)
        fitted_model = FitModel(fitted_node_tree, self.posterior)
        return fitted_model

    def survival_function(self, flat_parameter_array):
        param_dict = self.node_tree.unflatten_parameter_array(flat_parameter_array)

        def surv_function(**kwargs):
            return self.node_tree["y"].sf(**{**param_dict, **kwargs})

        return surv_function

    def sample_survival_function(self, n_samples):
        posterior_samples = self.posterior.sample(n_samples)[:n_samples]
        return [self.survival_function(x) for x in posterior_samples]

    def plot_survival_function(self):
        param_dict = self.node_tree.unflatten_parameter_array(self.posterior.maximum_likihood)
        start_point = self.node_tree["y"].ppf(0.005, **param_dict)
        end_point = self.node_tree["y"].ppf(0.995, **param_dict)
        x_s = np.linspace(start_point, end_point, 1000)
        surv_function = self.survival_function(self.posterior.maximum_likihood)
        vals = surv_function(y=x_s)
        from matplotlib import pyplot as plt
        plt.plot(x_s, vals)
