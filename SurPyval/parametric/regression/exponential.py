import numpy as np
import scipy.stats

from SurPyval.node.tree import NodeTree
from SurPyval.parameter.transformation import Transformation
from SurPyval.model.model import Model
from SurPyval.parameter.parameter import Parameter
from SurPyval.node.datalikihoodnode import DataLikihoodNode


class ExponentialRegression(Model):
    """
        Fit an exponential regression to the lifetime data with coefficients

        The linear predictor is currently transformed through exp(.) following XXX

        Parameters:
            * beta - the coefficients for the covariates

        Transformations:
            * alpha - (n * 1) parameters for individual scale of exponential distirbution

        Data:
            * x - an (n x k) matrix of covariates
            * event - an (n * 1) vector of int where `1` means an event happened and `0` means the observation was censored

        Nodes:
            * beta - prior distribution for beta (usually Gaussian)
            * y_event - likihood for non-censored observations
            * y_censored - likihood for censored observations

        Prior:
            * beta - prior for coefficient for covariates, commonly Gaussian

        $$\lambda $$
    """

    def __init__(self, prior_dict, y, event, x):

        self.parameters = [Parameter("beta", 1)]
        self.data_dict = {
            "x": x,
            "event": event
        }

        self.transformations = [
            Transformation(
                    lambda data_dict, parameter_dict: np.exp(np.sum(data_dict["x"] * parameter_dict["beta"], axis = 1)),
                    "alpha")
        ]
        
        self.node_dict = {
            "beta": prior_dict["beta"],
            "y": DataLikihoodNode(scipy.stats.expon, {"alpha": "scale"})
        }

        self.node_tree = NodeTree(self.node_dict, 
                                  self.data_dict, 
                                  self.parameters, 
                                  self.transformations)

    @staticmethod
    def show_plate():

        import matplotlib.pyplot as plt
        from matplotlib import rc
        import daft

        plt.rcParams['figure.figsize'] = 14, 8
        rc("font", family="serif", size=12)
        rc("text", usetex=False)

        pgm = daft.PGM(shape=[2.5, 3.5],
                       origin=[0, 0],
                       grid_unit=4,
                       label_params={'fontsize':18},
                       observed_style='shaded')

        pgm.add_node(daft.Node("beta", r"$\beta$", 1, 2.4, scale=2))
        pgm.add_node(daft.Node("mu_0", r"$\mu_0$", 0.8, 3, scale=2, fixed=True, offset=(0, 10)))
        pgm.add_node(daft.Node("sigma_0", r"$\sigma_0$", 1.2, 3, scale=2, fixed=True, offset=(0, 6)))
        pgm.add_node(daft.Node("y", r"$y_i$", 1, 0.9, scale=2, observed=True))
        pgm.add_node(daft.Node("x", r"$x_i$", 1.7, 1.6, scale=2, observed=True))
        pgm.add_plate(daft.Plate([0.5, 0.5, 1.5, 1.4], label=r"$i \in 1:N$", shift=-0.1))

        pgm.add_edge("mu_0", "beta")
        pgm.add_edge("sigma_0", "beta")
        pgm.add_edge("beta", "y")
        pgm.add_edge("x", "y")

        pgm.render()
        plt.show()
