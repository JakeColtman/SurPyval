from numpy.random import gamma


class Gamma:
    
    def __init__(self, alpha, llambda):
        self.alpha = alpha
        self.llambda = llambda

    def sample(self, n_samples):
        return gamma(self.alpha, 1. / self.llambda, n_samples)

def gamma_from_mean_variance(mean, variance):
    llambda = mean / variance
    alpha = llambda * mean
    return Gamma(alpha, llambda)