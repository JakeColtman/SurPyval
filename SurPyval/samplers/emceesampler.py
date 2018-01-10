
from SurPyval.samplers.sampler import Sampler

class EmceeSampler(Sampler):
    
    def __init__(self, sampler, pos):
        self.sampler = sampler
        self.pos = pos

    def sample(self, n_samples):
        self.sampler.reset()
        self.sampler.run_mcmc(self.pos, n_samples)
        return self.sampler.flatchain

