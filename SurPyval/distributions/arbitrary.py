import numpy as np

class ArbitraryDistribution(Distribution):
    
    def __init__(self, log_lik_f):
        self.log_lik_f = log_lik_f

    def log_lik(self, **kwargs):
        return self.log_lik_f(**kwargs)

    def pdf(self, **kwargs):
        return np.exp(self.log_lik(**kwargs))