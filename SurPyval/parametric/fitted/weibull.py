import numpy as np
import scipy.stats

from SurPyval.node import NodeTree, Node, DataNode, DeterministicNode, ParameterNode, DataLikihoodNode
from SurPyval.model.model import Model


class FittedWeibull(Model):
    """
        Fit an exponential regression to the lifetime data with coefficients

        The linear predictor is currently transofrmed through exp(.) following XXX

        Nodes:
            * llambda - the main parameter for the expoential distribution
            * y - predictive distribution for lifetime

        Likihood: 
            $$L(\lambda | D) = \prod f(y_i | \lambda)^{v_i} S(y_i | \lambda)^{1 - v_i}$$
            $$L(\lambda | D) = \lambda^{d} e^{-\lambda n \bar{y}}$$
    
        Prior:
            There isn't a (reasonable) conjugate prior for \beta, so a Gaussian prior is used
            This means that fitting the model requires some numerical approximation, currently MCMC
    """

    def __init__(self, y, event, alpha_prior, llambda_prior):

        node_dict = {
            "alpha": alpha_prior,
            "llambda": llambda_prior,
            "y": DataLikihoodNode(scipy.stats.weibull_min, y, {"alpha": "shape", "llambda": "scale", "y": "x"}),
            "event": DataNode("event", event)
        }

        node_tree = NodeTree(node_dict)

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
        
        pgm.add_node(daft.Node("lambda", r"$\lambda$", 1.5, 2.4, scale=2))
        pgm.add_node(daft.Node("mu_0", r"$\mu_0$", 1.3, 3, scale=2,
                            fixed=True, offset=(0,10)))
        pgm.add_node(daft.Node("sigma_0", r"$\sigma_0$", 1.7, 3, scale=2,
                            fixed=True, offset=(0,6)))
        
        pgm.add_node(daft.Node("y", r"$y_i$", 1, 1.4, scale=2, observed=True))
        pgm.add_plate(daft.Plate([0.5, 0.7, 1, 1.3], label=r"$i \in 1:N$", 
                                shift=-0.1))

        pgm.add_edge("alpha_0", "alpha")
        pgm.add_edge("kappa_0", "alpha")

        pgm.add_edge("sigma_0", "lambda")
        pgm.add_edge("mu_0", "lambda")
        
        pgm.add_edge("lambda", "y")
        pgm.add_edge("alpha", "y")


        pgm.render()
        plt.show()