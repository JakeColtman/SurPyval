from __future__ import division
import numpy as np
from SurPyval.distributions.gamma import Gamma   
from SurPyval.core.sampling import exp

class ConjugatePriorFittedExponential:

    def __init__(self, data, event):
        self.data = data
        self.event = event
        self.fit()

    def fit(self, gamma_prior):
        d = np.sum(self.event)
        y_sum = np.sum(self.data)
        self.fitted_l = Gamma(gamma_prior.alpha + d, gamma_prior.llambda + y_sum)
        return self

    def posterior_predictive(self):
        return exp(self.fitted_l)

    def survival_function(self, y):
        l_s = self.fitted_l * y
        return exp(l_s * -1)
    