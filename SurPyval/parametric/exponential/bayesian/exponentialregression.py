from SurPyval.distributions.exponential import Exponential
import emcee as em
import numpy as np

class PriorFittedExponentialRegression:

    def __init__(self, y_s, x_s, event):
        self.y_s = y_s
        self.x_s = x_s
        self.event = event

    @staticmethod
    def log_lik(beta, x_s, y_s, event):
        beta = beta * -1
        return np.dot(event , np.dot(x_s, beta)) - np.dot(y_s , np.exp(np.dot(x_s, beta)))

    def fit_guassian_prior(self, prior_means, prior_covar, n_walkers = 16, samples = 100):

        def lnprior(x):
            resid = (x - prior_means)
            numerator = - np.dot(resid.T, np.dot(np.linalg.inv(prior_covar), resid))
            denom = np.log(np.power(2.0 * np.pi, len(prior_means)) * np.linalg.det(prior_covar))
            return numerator - denom

        return self.fit(lnprior, n_walkers = n_walkers, samples = samples)

    def fit(self, prior_loglikihood, n_walkers = 16, samples = 1000, burn = 1000):
        
        def lnprob(beta, x_s, y_s, event):
            lp = prior_loglikihood(beta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + self.log_lik(beta, x_s, y_s, event)
        
        def generate_starting_points():
            neg_lok_lik = lambda *args: -self.log_lik(*args)
            result = op.minimize(neg_lok_lik, [1] * self.x_s.shape[1], args=(self.x_s, self.y_s, self.event))
            max_lik_point = result["x"]
            return [max_lik_point + np.random.normal(0, 0.01, self.x_s.shape[1]) for x in range(n_walkers)]
            
        ndim = self.x_s.shape[1]
        sampler = em.EnsembleSampler(n_walkers, ndim, lnprob, args=(self.x_s, self.y_s, self.event))
        p0 = generate_starting_points()
        pos, prob, state = sampler.run_mcmc(p0, burn)
        sampler.reset()
        sampler.run_mcmc(pos, samples)
        self.fitted_beta = sampler
        return self      