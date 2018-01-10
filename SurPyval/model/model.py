from scipy.optimize import minimize
import emcee as em
import numpy as np

from SurPyval.samplers.emceesampler import EmceeSampler


class Model(object):
    """
        High level class that does the actual number crunching
        Runs the MCMC approximation to the marginalization

        Thoughts for the future:
            * Allow non-emcee MCMC
            * Allow other approximations (VI?)
            * Support conjugate updating
    """
    
    def sample_posterior(self, n_samples, store = True, append = True):
        new_samples = self.posterior.sample(n_samples)
        if store:
            if append: 
                try:
                    self.posterior_samples.append(new_samples)
                except:
                    self.posterior_samples = [new_samples]
            else:
                self.posterior_samples = [new_samples]
                
    def fit(self, n_walkers = 4, burn = 500):

        def generate_starting_points():
            max_lik_point = self.maximum_likihood()
            return [max_lik_point + np.random.normal(0, 0.01, int(self.node_tree.length())) for x in range(n_walkers)]
        
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