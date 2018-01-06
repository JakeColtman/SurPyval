from numpy.random import gamma
import seaborn as sns
from matplotlib import pyplot as plt

from SurPyval.core.sampling import NumpySampler

class Gamma(NumpySampler):
    
    def __init__(self, alpha, llambda):
        self.alpha, self.llambda = alpha, llambda
        self.sampler = NumpySampler(gamma, shape = alpha, scale = 1./ llambda)

    def sample(self, n_samples):
        return self.sampler.sample(n_samples)

    def plot(self, n_samples = 10000):
        samples = self.sample(n_samples)
        sns.distplot(samples)
        plt.show()
        

def gamma_from_mean_variance(mean, variance):
    llambda = mean / variance
    alpha = llambda * mean
    return Gamma(alpha, llambda)