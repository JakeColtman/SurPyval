import numpy as np

from SurPyval.samplers.sampler import Sampler

class NumpySampler(Sampler):
    
    def __init__(self, np_random, **kwargs):
        self.np_random = np_random
        self.kwargs = kwargs
        
    def sample(self, n_samples):
        return self.np_random(size = n_samples, **self.kwargs).reshape(n_samples, 1)  
