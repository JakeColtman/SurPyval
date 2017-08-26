from SurPyval.distributions.exponential import Exponential
from SurPyval.core.sampling import NumpySampler

class MaximumLikihoodFittedExponentialRegression:

    def __init__(self, y_s, x_s, event):
        self.y_s = y_s
        self.x_s = x_s
        self.event = event
        self.fit()

    @staticmethod
    def log_lik(y_s, x_s, event, beta):
        llambda = np.dot(x_s.T, beta)
        return np.dot(event , np.dot(x_s.T, beta)) - np.dot(y_s , np.exp(np.dot(x_s.T, beta)))

    def fit(self, starting_point = None):
        def function_to_minimize(beta):
            return -1 * self.log_lik(self.y_s, self.x_s, self.event, beta)
        if starting_point is None:
            starting_point = tuple([0.5] * self.x_s.shape[0])
        result = minimize(function_to_minimize, starting_point)
        self.fitted_beta = result["x"]
        self.log_lik = self.log_lik(self.y_s, self.x_s, self.event, self.fitted_l)
        return self

    def sample_posterior_predictive(self, x, n_samples):
        llambda = np.dot(x_s.T, self.fitted_beta)
        return NumpySampler(np.random.exponential, scale = llambda)


