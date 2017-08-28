import numpy as np
from scipy.optimize import minimize

class PieceWiseConstantHazards:

    def __init__(self, y_s, x_s, event, period_lengths):
        self.y_s, self.x_s, self.event, self.period_lengths = y_s, x_s, event, period_lengths

    @staticmethod
    def lifetime_into_periods(period_lengths, lifetime):
        """
        
        >>> lifetime_in_periods(np.array([1., 1., 1.]), 2)
        array([ 1.,  1.,  0.])
        >>> lifetime_in_periods(np.array([1., 1., 1.]), 2.5)
        array([ 1.,  1.,  0.5])
        """
        periods_lived_in = []
        accounted_for_life = 0
        for period_length in period_lengths:
            if accounted_for_life + period_length > lifetime:
                periods_lived_in.append(lifetime - accounted_for_life)
                accounted_for_life = lifetime
                break
            else: 
                periods_lived_in.append(period_length)
                accounted_for_life += period_length
        else:
            raise ValueError("Longer lifetime than total period lengths")
            
        return np.array(periods_lived_in)

    @staticmethod
    def total_baseline_hazard(baseline_hazards, periods_lived_in):
        return np.dot(baseline_hazards[:len(periods_lived_in)], periods_lived_in)
            
    @staticmethod
    def log_lik_observation(y, x, beta, baseline_hazards, period_lengths, event):
        lifetime_in_periods = PieceWiseConstantHazards.lifetime_into_periods(period_lengths, y)
        death_period = len(lifetime_in_periods) - 1
        hazard_in_death_period = baseline_hazards[death_period]
        
        contribution_baseline_hazard = PieceWiseConstantHazards.total_baseline_hazard(baseline_hazards, lifetime_in_periods)
        
        contribution_survival = - (contribution_baseline_hazard * np.exp(np.dot(x, beta)))
        contirbution_death = np.log(hazard_in_death_period) + np.dot(x, beta)
        
        if event:
            return contirbution_death + contribution_survival
        else:
            return contribution_survival
        
    @staticmethod
    def log_lik(y_s, x_s, beta, baseline_hazards, period_lengths, event):
        log_lik_individuals = []
        for ii in range(len(y_s)):
            log_lik_ii = PieceWiseConstantHazards.log_lik_observation(y_s[ii], x_s.as_matrix()[ii], beta, baseline_hazards, period_lengths, event[ii][0])
            log_lik_individuals.append(log_lik_ii)
        return np.sum(log_lik_individuals)

    def fit(self):
        def function_to_minimize(params):
            num_periods = len(self.period_lengths)
            params = np.array(params)
            baseline_hazards = params[:num_periods]
            beta = params[num_periods:]
            return - self.log_lik(self.y_s, self.x_s, beta, baseline_hazards, self.period_lengths, self.event)

        num_periods = len(self.period_lengths)
        num_coeffs = self.x_s.shape[1]
        num_params = num_periods + num_coeffs

        starting_point = np.random.uniform(0, 1, num_params)
        hazard_bounds = [(0., 1.)] * num_periods 
        coeff_bounds = [(None, None)] * num_coeffs
        bounds = hazard_bounds + coeff_bounds

        result = minimize(function_to_minimize, starting_point, bounds = bounds, method = "SLSQP")
        self.result = result
        
        self.baseline_hazards = self.result["x"][:num_periods]
        self.fitted_beta = self.result["x"][num_periods:]
        return self