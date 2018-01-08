import numpy as np

from SurPyval.node.node import Node
from SurPyval.node.tree import NodeTree, Transformation
from SurPyval.model.model import Model
from SurPyval.parameter.parameter import Parameter
from SurPyval.distributions.datalikihood import DataLikihood

def likihood_distr(y, alpha_event, alpha_censored):
    return np.sum(np.log(alpha_event)) - np.dot(y.T , alpha_event)

def survival_distr(y, alpha_event, alpha_censored):
    return - np.dot(y.T , alpha_censored)

class LikihoodNode(Node):
    
    def __init__(self, y, event, x, parameter_dict = {"alpha_event": "alpha_event", "alpha_censored": "alpha_censored"}):
        self.parameter_names = parameter_dict.values()
        self.parameter_dict = parameter_dict
        self.distribution = DataLikihood(likihood_distr, survival_distr, y, event, x)

    def sample(self, x, n_samples):
        samples = np.exp(np.dot(x.T, self.distribution.sample(n_samples).T))
        prior_predictive_samples = np.array(map(lambda x: np.random.exponential(1.0 / x), samples))
        return prior_predictive_samples

class ExponentialRegression(Model):
    """
        Fit an exponential regression to the lifetime data with coefficients

        The linear predictor is currently transformed through exp(.) following XXX

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


    def __init__(self, prior_dict, y, event, x):

        self.parameters = [Parameter("beta", 1.0)]
        self.data_dict = {
            "x": x,
            "event": event
        }

        self.transformations = [
            Transformation( \
                    lambda data_dict, parameter_dict: np.exp(np.dot(data_dict["x"][data_dict["event"].astype(bool)], parameter_dict["beta"])),
                    "alpha_event"),
            Transformation( \
                    lambda data_dict, parameter_dict: np.exp(np.dot(data_dict["x"][~data_dict["event"].astype(bool)], parameter_dict["beta"])),
                    "alpha_censored")
        ]
        
        self.node_dict = {
            "beta": prior_dict["beta"],
            "y": LikihoodNode(y, event, x, {"alpha_event": "alpha_event", "alpha_censored": "alpha_censored"})
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

        pgm = daft.PGM(shape=[2.5, 3.5], origin=[0, 0], grid_unit=4,
                    label_params={'fontsize':18}, observed_style='shaded')

        pgm.add_node(daft.Node("beta", r"$\beta$", 1, 2.4, scale=2))
        pgm.add_node(daft.Node("mu_0", r"$\mu_0$", 0.8, 3, scale=2,
                            fixed=True, offset=(0,10)))
        pgm.add_node(daft.Node("sigma_0", r"$\sigma_0$", 1.2, 3, scale=2,
                            fixed=True, offset=(0,6)))
        pgm.add_node(daft.Node("y", r"$y_i$", 1, 0.9, scale=2, observed=True))
        pgm.add_node(daft.Node("x", r"$x_i$", 1.7, 1.6, scale=2, observed=True))
        pgm.add_plate(daft.Plate([0.5, 0.5, 1.5, 1.4], label=r"$i \in 1:N$", 
                                shift=-0.1))

        pgm.add_edge("mu_0", "beta")
        pgm.add_edge("sigma_0", "beta")
        pgm.add_edge("beta", "y")
        pgm.add_edge("x", "y")


        pgm.render()
        plt.show()