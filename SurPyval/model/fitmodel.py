from typing import Dict, Any

from SurPyval.node import NodeTree
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

    def predict(self, data_dict: Dict[str, Any]):
        fitted_node_tree = NodeTree(self.node_tree.node_dict, data_dict)
        fitted_model = FitModel(fitted_node_tree, self.posterior)
        return fitted_model

    def sample_survival(self):
        def plot_survival_function(surv_node):
            start_point = surv_node.distribution.ppf( 0.005, **parsed_n )[0]
            end_point = surv_node.distribution.ppf( 0.995, **parsed_n )[0]
            x_s = np.linspace( start_point, end_point, 1000 )
            vals = [surv_node.distribution.sf( x, **parsed_n )[0] for x in x_s]
            from matplotlib import pyplot as plt
            plt.plot( x_s, vals )