
import seaborn as sns

from SurPyval.distributions.gamma import Gamma
from SurPyval.core.sampling import NumpySampler


def create_prior_from_deaths_and_total_observed_time(deaths, total_observed_time):
    alpha = deaths
    llambda = total_observed_time
    return Gamma(alpha, llambda)

def create_prior_from_deaths_and_average_lifetime(deaths, average_lifetime):
    alpha = deaths
    llambda = average_lifetime * deaths
    return Gamma(alpha, llambda)

class DistributionNode:
    
    def __init__(self, name, prior_distribution, posterior_distribution):
        self.name = name
        self.prior = prior_distribution
        self.posterior = posterior_distribution

    def sample_prior(self, n_samples = 10000):
        return self.prior.sample(n_samples)

    def sample_posterior(self, n_samples = 10000):
        return self.posterior.sample(n_samples)

class PosteriorPredictiveDistribution(NumpySampler):

    def __init__(self, gamma_distribution):
        self.distr = gamma_distribution

    def sample(self, n_samples):
        samples = self.distr.sample(n_samples)
        posterior_predictive_samples = np.array(map(lambda x: np.random.exponential(1.0 / x), samples))
        return posterior_predictive_samples

    def plot(self, n_samples = 10000):
        sns.distplot(self.sample(n_samples))

class FittedExponential:
    """
        Fit an exponential distribution to the life lengths

        Nodes:
            * llambda - the main parameter for the expoential distribution
            * y - predictive distribution for lifetime

        Likihood: 
            $$L(\lambda | D) = \prod f(y_i | \lambda)^{v_i} S(y_i | \lambda)^{1 - v_i}$$
            $$L(\lambda | D) = \lambda^{d} e^{-\lambda n \bar{y}}$$
    
        Prior:
            defaults to the conjugate Gamma prior where:
                alpha => Number of observed deaths worth of prior
                lambda => Total observed time
    """


    def __init__(self, prior_dict, y, event):
        self.prior_dict = prior_dict
        self.d = np.sum(event)
        self.y_sum = np.sum(y)
        llambda_posterior = Gamma(prior.alpha + self.d, prior.llambda + self.y_sum)

        self.constants = {"alpha_0": prior_dict["llambda"].alpha, "llambda_0": prior_dict["llambda"].llambda}
        self.nodes = {
            "llambda": DistributionNode("llambda", prior_dict["llambda"], llambda_posterior),
            "y": DistributionNode("y", PosteriorPredictiveDistribution(prior_dict["llambda"]), PosteriorPredictiveDistribution(llambda_posterior))
        }

    def fit(self):
        pass

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

        pgm.add_node(daft.Node("lambda", r"$\lambda$", 1, 2.4, scale=2))
        pgm.add_node(daft.Node("alpha_0", r"$\alpha_0$", 0.8, 3, scale=2,
                                fixed=True, offset=(0,10)))
        pgm.add_node(daft.Node("lambda_0", r"$\lambda_0$", 1.2, 3, scale=2,
                                fixed=True, offset=(0,6)))
        pgm.add_node(daft.Node("y", r"$y_i$", 1, 1.4, scale=2, observed=True))
        pgm.add_plate(daft.Plate([0.5, 0.7, 1, 1.3], label=r"$i \in 1:N$", 
                                    shift=-0.1))

        pgm.add_edge("alpha_0", "lambda")
        pgm.add_edge("lambda_0", "lambda")
        pgm.add_edge("lambda", "y")

        pgm.render()
        plt.show()