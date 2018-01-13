import numpy as np
import scipy.stats

from SurPyval.node.tree import NodeTree
from SurPyval.node.node import Node
from SurPyval.node.transformation import DeterministicNode
from SurPyval.model.model import Model
from SurPyval.node.parameter import ParameterNode
from SurPyval.node.datalikihoodnode import DataLikihoodNode


class WeibullRegression(Model):
    """
        Fit an weibull regression to the lifetime data with covariates

        Covariates enter the likihood through the scale parameter of the Weibull distribution
        i.e. all covariates have the same shape, but can have varying scales

        Parameters
        ----------
        y: array-like
           (n * 1) array of lifetimes - both censored and events
        event: array-like
            (n * 1) array of whether an event was observed
            1 if event occurred or 0 if observation is censored
        x: array-like
           (n * k) array of covariates
        alpha_prior: Node
                    Commonly a Gaussian node
        beta_prior: Node
                    Commonly a Gaussian node
    """

    def __init__(self, y, event, x, alpha_prior, beta_prior):
        data_dict = {
            "y": y,
            "x": x,
            "event": event
        }

        llambda = DeterministicNode(
                lambda data_dict, parameter_dict: np.sum(data_dict["x"] * parameter_dict["beta"], axis=1),
                "llambda")

        node_dict = {
            "beta": ParameterNode(beta_prior, "beta", 1),
            "alpha": ParameterNode(alpha_prior, "alpha", 1),
            "llambda": llambda,
            "y": DataLikihoodNode(scipy.stats.weibull_min, {"alpha": "c", "llambda": "scale"})
        }

        node_tree = NodeTree(node_dict,
                             data_dict)

        Model.__init__(self, node_tree)

    @staticmethod
    def show_plate():
        import matplotlib.pyplot as plt
        from matplotlib import rc
        import daft

        plt.rcParams['figure.figsize'] = 14, 8
        rc("font", family="serif", size=12)
        rc("text", usetex=False)

        pgm = daft.PGM(shape=[2.5, 3.5], origin=[0, 0], grid_unit=4,
                    label_params={'fontsize':18}, observed_style='shaded')

        pgm.add_node(daft.Node("alpha", r"$\alpha$", 0.5, 2.4, scale=2))
        pgm.add_node(daft.Node("alpha_0", r"$\alpha_0$", 0.3, 3, scale=2,
                            fixed=True, offset=(0,10)))
        pgm.add_node(daft.Node("kappa_0", r"$\kappa_0$", 0.7, 3, scale=2,
                            fixed=True, offset=(0,6)))

        pgm.add_node(daft.Node("beta", r"$\beta$", 1.5, 2.4, scale=2))
        pgm.add_node(daft.Node("mu_0", r"$\mu_0$", 1.3, 3, scale=2,
                            fixed=True, offset=(0,10)))
        pgm.add_node(daft.Node("sigma_0", r"$\sigma_0$", 1.7, 3, scale=2,
                            fixed=True, offset=(0,6)))

        pgm.add_node(daft.Node("lambda", r"$\lambda_i$", 1.4, 1.2, scale=2))

        pgm.add_node(daft.Node("x", r"$x_i$", 1.9, 1.7, scale=2, observed=True))

        pgm.add_node(daft.Node("y", r"$y_i$", 1, 0.5, scale=2, observed=True))

        pgm.add_plate(daft.Plate([0.5, 0.2, 1.8, 1.8], label=r"$i \in 1:N$",
                                shift=-0.1))

        pgm.add_edge("alpha_0", "alpha")
        pgm.add_edge("kappa_0", "alpha")

        pgm.add_edge("sigma_0", "beta")
        pgm.add_edge("mu_0", "beta")

        pgm.add_edge("beta", "lambda")
        pgm.add_edge("x", "lambda")

        pgm.add_edge("lambda", "y")

        pgm.add_edge("alpha", "y")


        pgm.render()
        plt.show()