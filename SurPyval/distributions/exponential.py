from numpy.random import exponential


class Exponential:
    
    def __init__(self, alpha):
        self.alpha = alpha

    def sample(self, n_samples):
        return exponential(alpha, n_samples)