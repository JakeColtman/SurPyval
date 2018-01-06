
import seaborn as sns
import emcee as em
from scipy.optimize import minimize
import emcee as em

from SurPyval.core.sampling import EmceeSampler
from SurPyval.distributions.gamma import Gamma
from SurPyval.core.sampling import NumpySampler

class DistributionNode:
    
    def __init__(self, name, prior_distribution, posterior_distribution):
        self.name = name
        self.prior = prior_distribution
        self.posterior = posterior_distribution

    def sample_prior(self, n_samples = 10000):
        return self.prior.sample(n_samples)

    def sample_posterior(self, n_samples = 10000):
        return self.posterior.sample(n_samples)

class PriorPredictiveDistribution(NumpySampler):
    
    def __init__(self, gaussian_distribution):
        self.distr = gaussian_distribution

    def sample(self, x, n_samples):
        samples = np.exp(np.dot(x.T, self.distr.sample(n_samples).T))
        prior_predictive_samples = np.array(map(lambda x: np.random.exponential(1.0 / x), samples))
        return prior_predictive_samples

    def plot(self, n_samples = 10000):
        sns.distplot(self.sample(n_samples))

class ExponentialRegression:
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


    def __init__(self, prior_dict, y, event, x):
        self.constants = {"mu_0": prior_dict["beta"].mu, "var_0": prior_dict["beta"].covar}
        self.data = {
            "y": y,
            "x": x,
            "event": event,
            "d": np.sum(event),
            "y_sum": np.sum(y)
        }
        self.nodes = {
            "beta": DistributionNode("beta", prior_dict["beta"], None),
            "y": DistributionNode("y", PriorPredictiveDistribution(prior_dict["beta"]), None)
        }

    def fit(self, n_walkers = 4, burn = 500):

        def generate_starting_points():
            max_lik_point = self.maximum_likihood()
            return [max_lik_point + np.random.normal(0, 0.01, self.data["x"].shape[1]) for x in range(n_walkers)]
        
        ndim = self.data["x"].shape[1]
        sampler = em.EnsembleSampler(n_walkers, ndim, self.log_likihood)
        p0 = generate_starting_points()
        pos, prob, state = sampler.run_mcmc(p0, burn)
        
        self.nodes["beta"].posterior = EmceeSampler(sampler, pos)
        self.nodes["y"].posterior = PriorPredictiveDistribution(self.nodes["beta"].posterior)
        return self      

    def maximum_likihood(self):
        neg_lok_lik = lambda *args: -self.log_likihood(*args)
        result = minimize(neg_lok_lik, [1] * self.data["x"].shape[1])
        max_lik_point = result["x"]
        return max_lik_point
    
    def log_likihood(self, beta):
        def log_prior():
            resid = (beta - self.constants["mu_0"])
            numerator = - np.dot(resid.T, np.dot(np.linalg.inv(self.constants["var_0"]), resid))
            denom = np.log(np.power(2.0 * np.pi, len(beta)) * np.linalg.det(self.constants["var_0"]))
            return numerator - denom

        def log_lik():
            return np.dot(self.data["event"].T, np.dot(self.data["x"], beta)) - np.dot(self.data["y"].T , np.exp(np.dot(self.data["x"], beta)))

        return log_lik() + log_prior()

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