import numpy as np
import scipy.stats

from SurPyval.node.tree import NodeTree
from SurPyval.node.node import Node
from SurPyval.parameter.transformation import Transformation
from SurPyval.model.model import Model
from SurPyval.parameter.parameter import Parameter
from SurPyval.node.datalikihoodnode import DataLikihoodNode


class ExponentialRegression(Model):
    """
        Fit an exponential regression to the lifetime data with coefficients

        The linear predictor is (by default) transformed through exp(.) following XXX

        Parameters
        ----------
        y: array-like
           (n * 1) array of lifetimes - both censored and events
        event: array-like
            (n * 1) array of whether an event was observed
            1 if event occurred or 0 if observation is censored
        x: array-like
           (n * k) array of covariates
        beta_prior: Node
                    Commonly a Gaussian node

        Examples
        --------
        >>> from SurPyval.node.node import gaussian
        >>> import pandas as pd
        >>> beta_prior = gaussian({"beta": "x"}, constant_dict={"loc": 0.0, "scale": 100.0})
        >>> data = pd.DataFrame({"y": [1.0, 2.0], "x_0": [1.0, 1.0], "event": [1, 0]})
        >>> ExponentialRegression(data["y"], data["event"], data["x"], beta_prior})
    """

    def __init__(self, y, event, x, beta_prior):

        data_dict = {
            "y": y,
            "x": x,
            "event": event
        }

        transformations = [
            Transformation(
                lambda data_dict, parameter_dict: np.exp( np.sum( data_dict["x"] * parameter_dict["beta"], axis=1)),
                "alpha" )
        ]

        node_dict = {
            "beta": beta_prior,
            "y": DataLikihoodNode(scipy.stats.expon, {"alpha": "scale"})
        }
        parameters = [
            Parameter("beta", 1)
        ]
        node_tree = NodeTree(node_dict,
                             data_dict,
                             parameters,
                             transformations)

        Model.__init__(self, node_tree)

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
