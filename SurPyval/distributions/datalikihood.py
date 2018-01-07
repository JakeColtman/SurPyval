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
        likihood_contribution = np.dot(self.event.T, self.likihood_dist(self.y, self. x, **kwargs))
        survival_contribution = np.dot(self.censored.T, self.survival_dist(self.y, self.x, **kwargs))
        return likihood_contribution + survival_contribution

    def pdf(self, **kwargs):
        return np.exp(self.log_lik(**kwargs))

    