import numpy as np
import emcee as em

from SurPyval.samplers.emceesampler import EmceeSampler
from SurPyval.distributions.distribution import Distribution

class DataLikihood(Distribution):

    def __init__(self, likihood_dist, survival_dist, y, event, x):
        self.event = event
        self.censored = 1.0 - self.event
        self.y = y
        self.x = x
        self.likihood_dist = likihood_dist
        self.survival_dist = survival_dist

    def log_lik(self, **kwargs):
        likihood_contribution = self.likihood_dist(self.y[self.event.astype(bool)], self.x[self.event.astype(bool)], **kwargs)
        survival_contribution = self.survival_dist(self.y[self.censored.astype(bool)], self.x[self.censored.astype(bool)], **kwargs)
        return likihood_contribution + survival_contribution

    def pdf(self, **kwargs):
        return np.exp(self.log_lik(**kwargs))

    def survival_sampler(self, n_walkers, burn, **kwargs):
        ndim = 1
        sampler = em.EnsembleSampler(2, ndim, self.survival_log_lik, kwargs = kwargs)
        p0 = [1.0 + np.random.normal(0, 0.01, 1) for x in range(n_walkers)]
        pos, prob, state = sampler.run_mcmc(p0, 500)

        return EmceeSampler(sampler, pos)

    def survival_log_lik(self, y, **kwargs):
        if y <= 0:
            return -np.inf
        return self.survival_dist(y = y, **kwargs)