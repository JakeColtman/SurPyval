from numpy.random import multivariate_normal


class Gaussian:
    
    def __init__(self, mu, covar):
        if type(covar) is float:
            covar = np.array([[covar]])
        if type(mu) is float:
            mu = np.array([mu])
        self.mu, self.covar = mu, covar

    def sample(self, n_samples):
        return multivariate_normal(self.mu, self.covar, n_samples)