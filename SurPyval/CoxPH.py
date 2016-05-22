import numpy as np
from numpy import dot, exp
from numpy.linalg import solve, norm


class CoxPH:

    def __init__(self, df, duration_col, event_col):
        self.df = df.copy()
        self.df.sort_values(by=duration_col, inplace=True)
        self.duration_col = duration_col
        self.event_col = event_col
        self.preprocess()

    def preprocess(self):
        self.E = self.df[self.event_col]
        del self.df[self.event_col]
        self.T = self.df[self.duration_col]
        del self.df[self.duration_col]
        self.E = self.E.astype(bool)

    def fit(self):
        self.hazards_ = self._newton_rhaphson(self.df, self.T, self.E)

    def _get_efron_values(self, X, beta, T, E, include_likelihood=False):
        """
        Calculates the first and second order vector differentials,
        with respect to beta. If 'include_likelihood' is True, then
        the log likelihood is also calculated. This is omitted by default
        to speed up the fit.
        Note that X, T, E are assumed to be sorted on T!
        Parameters:
            X: (n,d) numpy array of observations.
            beta: (1, d) numpy array of coefficients.
            T: (n) numpy array representing observed durations.
            E: (n) numpy array representing death events.
        Returns:
            hessian: (d, d) numpy array,
            gradient: (1, d) numpy array
            log_likelihood: double, if include_likelihood=True
        """

        n, d = X.shape
        hessian = np.zeros((d, d))
        gradient = np.zeros((1, d))
        log_lik = 0

        # Init risk and tie sums to zero
        x_tie_sum = np.zeros((1, d))
        risk_phi, tie_phi = 0, 0
        risk_phi_x, tie_phi_x = np.zeros((1, d)), np.zeros((1, d))
        risk_phi_x_x, tie_phi_x_x = np.zeros((d, d)), np.zeros((d, d))

        # Init number of ties
        tie_count = 0

        # Iterate backwards to utilize recursive relationship
        for i, (ti, ei) in reversed(list(enumerate(zip(T, E)))):
            # Doing it like this to preserve shape
            xi = X[i:i + 1]
            # Calculate phi values
            phi_i = exp(dot(xi, beta))
            phi_x_i = dot(phi_i, xi)
            phi_x_x_i = dot(xi.T, xi) * phi_i

            # Calculate sums of Risk set
            risk_phi += phi_i
            risk_phi_x += phi_x_i
            risk_phi_x_x += phi_x_x_i
            # Calculate sums of Ties, if this is an event
            if ei:
                x_tie_sum += xi
                tie_phi += phi_i
                tie_phi_x += phi_x_i
                tie_phi_x_x += phi_x_x_i

                # Keep track of count
                tie_count += 1

            if i > 0 and T[i - 1] == ti:
                # There are more ties/members of the risk set
                continue
            elif tie_count == 0:
                # Only censored with current time, move on
                continue

            # There was atleast one event and no more ties remain. Time to sum.
            partial_gradient = np.zeros((1, d))

            for l in range(tie_count):
                c = l / tie_count

                denom = (risk_phi - c * tie_phi)
                z = (risk_phi_x - c * tie_phi_x)

                if denom == 0:
                    # Can't divide by zero
                    raise ValueError("Denominator was zero")

                # Gradient
                partial_gradient += z / denom
                # Hessian
                a1 = (risk_phi_x_x - c * tie_phi_x_x) / denom
                # In case z and denom both are really small numbers,
                # make sure to do division before multiplications
                a2 = dot(z.T / denom, z / denom)

                hessian -= (a1 - a2)

                if include_likelihood:
                    log_lik -= np.log(denom).ravel()[0]

            # Values outside tie sum
            gradient += x_tie_sum - partial_gradient
            if include_likelihood:
                log_lik += dot(x_tie_sum, beta).ravel()[0]

            # reset tie values
            tie_count = 0
            x_tie_sum = np.zeros((1, d))
            tie_phi = 0
            tie_phi_x = np.zeros((1, d))
            tie_phi_x_x = np.zeros((d, d))

        if include_likelihood:
            return hessian, gradient, log_lik
        else:
            return hessian, gradient

    def _newton_rhaphson(self, X, T, E, initial_beta=None, step_size=1.,
                         precision=10e-5, show_progress=True, include_likelihood=False):
        """
        Newton Rhaphson algorithm for fitting CPH model.
        Note that data is assumed to be sorted on T!
        Parameters:
            X: (n,d) Pandas DataFrame of observations.
            T: (n) Pandas Series representing observed durations.
            E: (n) Pandas Series representing death events.
            initial_beta: (1,d) numpy array of initial starting point for
                          NR algorithm. Default 0.
            step_size: float > 0.001 to determine a starting step size in NR algorithm.
            precision: the convergence halts if the norm of delta between
                     successive positions is less than epsilon.
            include_likelihood: saves the final log-likelihood to the CoxPHFitter under _log_likelihood.
        Returns:
            beta: (1,d) numpy array.
        """
        assert precision <= 1., "precision must be less than or equal to 1."
        n, d = X.shape

        # Want as bools
        E = E.astype(bool)

        # make sure betas are correct size.
        if initial_beta is not None:
            assert initial_beta.shape == (d, 1)
            beta = initial_beta
        else:
            beta = np.zeros((d, 1))

        get_gradients = self._get_efron_values

        i = 1
        converging = True
        # 50 iterations steps with N-R is a lot.
        # Expected convergence is ~10 steps
        while i < 50:# and step_size > 0.001:

            output = get_gradients(X.values, beta, T.values, E.values, include_likelihood=include_likelihood)
            h, g = output[:2]

            delta = solve(-h, step_size * g.T)
            if np.any(np.isnan(delta)):
                raise ValueError("delta contains nan value(s). Convergence halted.")

            # Save these as pending result
            hessian, gradient = h, g

            if norm(delta) < precision:
                converging = False

            # Only allow small steps
            if norm(delta) > 10:
                step_size *= 0.5
                continue

            beta += delta

            if ((i % 10) == 0) and show_progress:
                print("Iteration %d: delta = %.5f" % (i, norm(delta)))
            i += 1

        self._hessian_ = hessian
        self._score_ = gradient
        if include_likelihood:
            self._log_likelihood = output[-1] if self.strata is None else ll
        if show_progress:
            print("Convergence completed after %d iterations." % (i))
        return beta

from lifelines.datasets import load_rossi
rossi_dataset = load_rossi()
cf = CoxPH(rossi_dataset, "week", "arrest")
cf.fit()
print(cf.hazards_)
