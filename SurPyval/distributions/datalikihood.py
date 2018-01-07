import numpy as np

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

    