import numpy as np

from SurPyval.distributions.distribution import Distribution


class DataLikihood(Distribution):

    def __init__(self, f_dist, y, event, x):
        self.event = event
        self.censored = 1.0 - self.event
        self.y = y
        self.x = x
        self.f_dist = f_dist

    def log_lik(self, **kwargs):
        likihood_contribution = self.f_dist.pdf(self.y[self.event.astype(bool)], **kwargs)
        survival_contribution = self.f_dist.sf(self.y[self.censored.astype(bool)], **kwargs)
        return np.sum(likihood_contribution) + np.sum(survival_contribution)

    def pdf(self, **kwargs):
        return np.exp(self.log_lik(**kwargs))

    def sample(self, **kwargs):
        u = np.random.uniform(0, 1, 1)
        return self.f_dist.ppf(u, **kwargs)