from scipy.optimize import minimize
import emcee as em
import numpy as np

from SurPyval.node.tree import NodeTree
from SurPyval.samplers.emceesampler import EmceeSampler


class Model(object):
    """
        High level class that coordinates the forming and estimating models

        Parameters
        ----------
        node_tree: NodeTree
                   The graphical structure of the model
    """

    def __init__(self, node_tree: NodeTree):
        self.node_tree = node_tree

    def fit(self, n_walkers: int =4, burn: int =500):
        """
        Generate a sampler for the posterior distribution

        Populate the `self.posterior` variable with a `Sampler` object.
        Sampling is started in a small sphere around the maximum likihood estimate

        Parameters
        ----------
        n_walkers: int, optional
                   number of MCMC chains to use (default 4)
        burn: int, option
              number of samples required to reach convergence (default 500)

        Returns
        -------
        None
            The function has the side effect of populating self.posterior
        """

        def generate_starting_points():
            max_lik_point = self.maximum_likihood()
            return [max_lik_point + np.random.normal(0, 0.01, int(self.node_tree.length())) for x in range(n_walkers)]
        
        ndim = self.node_tree.length()
        sampler = em.EnsembleSampler(n_walkers, ndim, self.node_tree.logpdf)
        p0 = generate_starting_points()
        pos, prob, state = sampler.run_mcmc(p0, burn)
        
        self.posterior = EmceeSampler(sampler, pos)
        return self      

    def maximum_likihood(self):
        """
        Calculate the parameter array that maximizes the joint likihood

        Corresponds to the the mode of samples from the posterior
        Useful for quick estimates and as a starting point for MCMC

        Returns
        -------
        array_like
            flat array of parameters
        """
        neg_lok_lik = lambda *args: -self.node_tree.logpdf(*args)
        result = minimize(neg_lok_lik, np.array([1] * self.node_tree.length()))
        max_lik_point = result["x"]
        return max_lik_point