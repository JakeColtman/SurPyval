from scipy.optimize import minimize
import emcee as em
import numpy as np

class Model(object):

    def __init__(self, node_tree):
        self.node_tree = node_tree

    def fit(self, n_walkers = 4, burn = 500):

        def generate_starting_points():
            max_lik_point = self.maximum_likihood()
            return [max_lik_point + np.random.normal(0, 0.01, self.node_tree.length()) for x in range(n_walkers)]
        
        ndim = self.node_tree.length()
        sampler = em.EnsembleSampler(n_walkers, ndim, self.node_tree.log_lik)
        p0 = generate_starting_points()
        pos, prob, state = sampler.run_mcmc(p0, burn)
        
        self.posterior = EmceeSampler(sampler, pos)
        return self      

    def maximum_likihood(self):
        neg_lok_lik = lambda *args: -self.node_tree.log_lik(*args)
        result = minimize(neg_lok_lik, [1] * self.node_tree.length())
        max_lik_point = result["x"]
        return max_lik_point