import emcee as em
import numpy as np

from SurPyval.samplers.sampler import Sampler


class EmceeSampler(Sampler):

    def __init__(self, lik_function, maximum_likihood, n_walkers=4):
        self.lik_function = lik_function
        self.n_walkers = n_walkers
        self.maximum_likihood = maximum_likihood
        self.pos = self.generate_starting_points()
        self.sampler = em.EnsembleSampler(n_walkers, len(self.maximum_likihood), self.lik_function)

    def sample(self, n_samples):
        self.sampler.reset()
        self.pos, prob, state = self.sampler.run_mcmc(self.pos, n_samples)
        return self.sampler.flatchain

    def generate_starting_points(self):
        max_lik_point = self.maximum_likihood
        return [max_lik_point + np.random.normal(0, 0.01, len(self.maximum_likihood)) for x in range(self.n_walkers)]
