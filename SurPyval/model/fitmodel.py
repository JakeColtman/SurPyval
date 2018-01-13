from SurPyval.model import Model
from SurPyval.node import NodeTree
from SurPyval.samplers import EmceeSampler


class FitModel(Model):
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

    def predict(self, data_dict):
        fitted_node_tree = NodeTree(self.node_tree.node_dict, data_dict, self.node_tree.parameters, self.node_tree.transformations)
        fitted_model = Model(fitted_node_tree)
        fitted_model.posterior = self.posterior
        return fitted_model
