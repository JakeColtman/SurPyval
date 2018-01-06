
import seaborn as sns
import emcee as em
from scipy.optimize import minimize

from SurPyval.core.sampling import Sampler
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
    
    def __init__(self, alpha_dist, llambda_dist):
        self.alpha_dist = alpha_dist
        self.llambda_dist = llambda_dist

    def sample(self, n_samples):
        alpha_samples = self.alpha_dist.sample(n_samples)
        llambda_samples = self.llambda_dist.sample(n_samples)
        samples = zip(alpha_samples, llambda_samples)
        prior_predictive_samples = np.array(map(lambda x: np.random.weibull(x[0], 1.0 / np.log(x)), samples))
        return prior_predictive_samples

    def plot(self, n_samples = 10000):
        sns.distplot(self.sample(n_samples))

class FittedWeibull:
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
        self.constants = {
                "mu_0": prior_dict["llambda"].mu, 
                "var_0": prior_dict["llambda"].covar,
                "alpha_0": prior_dict["alpha"].alpha,
                "kappa_0": prior_dict["alpha"].llambda
                }

        self.data = {
            "y": y,
            "event": event,
            "d": np.sum(event),
            "y_sum": np.sum(y),
            "x": x
        }
        self.nodes = {
            "llambda": DistributionNode("llambda", prior_dict["llambda"], None),
            "beta": DistributionNode("beta", prior_dict["beta"], None),
            "alpha": DistributionNode("alpha", prior_dict["alpha"], None),
            "y": DistributionNode("y", PriorPredictiveDistribution(prior_dict["alpha"], prior_dict["llambda"]), None)
        }

    def fit(self, n_walkers = 4, burn = 500):
    
        def generate_starting_points():
            max_lik_point = self.maximum_likihood()
            return [max_lik_point + np.random.normal(0, 0.01, 2) for x in range(n_walkers)]
        
        ndim = 2
        sampler = em.EnsembleSampler(n_walkers, ndim, self.log_likihood_flat)
        p0 = generate_starting_points()
        pos, prob, state = sampler.run_mcmc(p0, burn)
        sampler = EmceeSampler(sampler, pos)
        self.nodes["llambda"].posterior = Sampler.apply_to_sampler(lambda x: x[:,1], sampler)
        self.nodes["alpha"].posterior = Sampler.apply_to_sampler(lambda x: x[:,0], sampler)
        self.nodes["y"].posterior = PriorPredictiveDistribution(self.nodes["alpha"].posterior, self.nodes["llambda"].posterior)
        return self      

    def maximum_likihood(self):
        neg_log_lik = lambda *args: -self.log_likihood_flat(*args)
        result = minimize(neg_log_lik, [1, 1])
        max_lik_point = result["x"]
        return max_lik_point

    def log_likihood(self, alpha, llambda):
        
        def log_llambda_prior():
            resid = (llambda - self.constants["mu_0"])
            numerator = - np.dot(resid.T, np.dot(np.linalg.inv(self.constants["var_0"]), resid))
            denom = np.log(np.power(2.0 * np.pi, len(beta)) * np.linalg.det(self.constants["var_0"]))
            return numerator - denom

        def log_alpha_prior():
            import scipy.stats
            return np.log(scipy.stats.gamma.pdf(alpha, self.constants["alpha_0"], scale = self.constants["kappa_0"])) 

        def log_lik():
            if alpha <= 0 or llambda <= 0:
                return - np.inf
            return self.data["d"] * np.log(alpha) + self.data["d"] * llambda + (alpha - 1) * np.sum(self.data["event"] * np.log(self.data["y"])) - np.sum(np.exp(llambda) * self.data["y"] ** alpha)

        return log_lik() + log_alpha_prior() + log_llambda_prior()

    def flatten_parameters(self, alpha, llambda):
        return np.array([alpha, llambda])

    def unflatten_parameters(self, parameters):
        return {"alpha": parameters[0], "llambda": parameters[1]}

    def log_likihood_flat(self, parameters):
        return self.log_likihood(**self.unflatten_parameters(parameters))

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