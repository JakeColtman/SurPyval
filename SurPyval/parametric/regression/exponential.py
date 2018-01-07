from SurPyval.node.node import Node
from SurPyval.node.tree import NodeTree
from SurPyval.model.model import Model
from SurPyval.parameter.parameter import Parameter
from SurPyval.distributions.datalikihood import DataLikihood

def likihood_distr(y, x, beta):
    return len(y) * np.dot(x, beta) - np.dot(y.T , np.exp(np.dot(x, beta)))

def survival_distr(y, x, beta):
    return - np.dot(y.T , np.exp(np.dot(x, beta)))

# ExponentialRegressionDataLikihood = Data

# class DataLikihoodDistribution(Distribution):
    
#     def __init__(self, y, event, x):
#         self.y = y
#         self.event = event
#         self.x = x

#     def log_lik(self, beta):
#         return np.dot(self.event.T, np.dot(self.x, beta)) - np.dot(self.y.T , np.exp(np.dot(self.x, beta)))

class LikihoodNode(Node):
    
    def __init__(self, y, event, x, parameter_names = ["beta"]):
        self.parameter_names = parameter_names
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
        self.node_dict = {
            "beta": prior_dict["beta"],
            "y": LikihoodNode(y, event, x, ["beta"])
        }
        self.node_tree = NodeTree(self.node_dict, self.parameters)

    @staticmethod
    def show_plate():
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