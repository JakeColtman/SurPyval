import numpy as np

class Sampler:
    
    def __init__(self, sample_function):
        self.sample_function = sample_function
        
    def __add__(self, other):
        return self.combine_samplers(lambda a, b: a + b, self, other)
    
    def __sub__(self, other):
        return self.combine_samplers(lambda a, b: a - b, self, other)

    def __mul__(self, other):
        return self.apply_to_sampler(lambda a: a * other, self)
        
    def sample(self, n_samples):
        return self.sample_function(n_samples)
    
    @staticmethod
    def combine_samplers(f, x, y):
        def combined(f, x, y, n_samples):
            if isinstance(x, Sampler):
                x = x.sample(n_samples)
            if isinstance(y, Sampler):
                y = y.sample(n_samples)

            return f(x, y)
        return Sampler(lambda n_samples: combined(f, x, y, n_samples))
    
    @staticmethod
    def apply_to_sampler(f, x):
        return Sampler(lambda n_samples: f(x.sample(n_samples)))

class NumpySampler(Sampler):
    
    def __init__(self, np_random, **kwargs):
        self.np_random = np_random
        self.kwargs = kwargs
        
    def sample(self, n_samples):
        return self.np_random(size = n_samples, **self.kwargs).reshape(n_samples, 1)  

class EmceeSampler(Sampler):

    def __init__(self, sampler, pos):
        self.sampler = sampler
        self.pos = pos

    def sample(self, n_samples):
        self.sampler.reset()
        self.sampler.run_mcmc(self.pos, n_samples)
        return self.sampler.flatchain

def dot(x, y):
    return Sampler.combine_samplers(np.dot, x, y)

def exp(x):
    return Sampler.apply_to_sampler(np.exp, x)