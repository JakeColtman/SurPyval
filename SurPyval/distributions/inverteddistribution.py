from SurPyval.distributions.distribution import Distribution

class InvertedDistribution(Distribution):

    def __init__(self, distribution, **kwargs):
        self.distribution_class = distribution.__class__
        self.distribution_kwargs = kwargs

    def log_lik(self, **kwargs):
        return self.distribution_class(**kwargs).log_lik(**self.distribution_kwargs) 

    def pdf(self, **kwargs):
        return self.distribution_class(**kwargs).pdf(**self.distribution_kwargs) 