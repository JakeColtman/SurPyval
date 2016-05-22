import numpy as np
from numpy import dot, exp, transpose
from numpy.linalg import solve, norm
import numpy as np

class JakeCoxPH:

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

    def fit(self, start_beta = 0):
        data = self.df.values
        n, d = data.shape
        def calculate_gradients_for_coefficient(beta):

            self.risk_theta = 0
            self.risk_theta_x = np.zeros((1, d))
            self.risk_theta_x_t = np.zeros((d, 1))
            self.risk_theta_x_x = np.zeros((d,d))

            hessian = np.zeros((d, d))
            score = np.zeros((1, d))
            def theta(x_i):
                return exp(dot( x_i, beta))

            def add_to_risk_sums(x_i):
                self.risk_theta += theta(x_i)
                self.risk_theta_x += dot(theta(x_i), x_i)
                self.risk_theta_x_t += dot(theta(x_i) , x_i.T)
                self.risk_theta_x_x += theta(x_i) * dot(x_i.T, x_i)
                print("risks")
                print(self.risk_theta)
                print(self.risk_theta_x)

            for i, (ti, ei) in reversed(list(enumerate(zip(self.T, self.E)))):
                x_i = data[i: i + 1]
                print(ti)
                add_to_risk_sums(x_i)

                if ei:
                    score += (x_i - (self.risk_theta_x/self.risk_theta))
                    print("score", score)
                    hessian -= (self.risk_theta_x_x / self.risk_theta_x) + (dot(self.risk_theta_x, self.risk_theta_x_t) / self.risk_theta ** 2)

            return score, hessian

        def newton_rhaphson(beta, step_size=1.,
                             precision=10e-5, show_progress=True):

            n, d = self.df.shape
            E = self.E.astype(bool)

            get_gradients = calculate_gradients_for_coefficient

            i = 1
            converging = True
            while converging and i < 20 and step_size > 0.001:
                print(i)
                output = get_gradients(beta)
                g, h = output[:2]
                print(g)
                delta = solve(-h, step_size * g.T)
                hessian, gradient = h, g

                if norm(delta) < precision:
                    converging = False

                # Only allow small steps
                if norm(delta) > 10:
                    step_size *= 0.5
                    continue

                beta += delta
                print(beta)
                i += 1

            self._hessian_ = hessian
            self._score_ = gradient
            return beta

        print(newton_rhaphson(start_beta))


from lifelines.datasets import load_rossi
times = [1, 4, 7, 12, 16]
events = [1,1,1,1,1]
covariates = [1.3,2.1,1.5,3.4,1.2]
covariatesTwo = [1, 1, 1, 2, 2]
import pandas as pd
df = pd.DataFrame()
df["times"] = times
df["events"] = events
df["x"] = covariates

#
from lifelines import CoxPHFitter
ph = CoxPHFitter()
ph.fit(df, "times", "events", initial_beta = np.array([[0.2]]))
print(ph.hazards_)
#
#
# ph = JakeCoxPH(df, "times", "events")
# ph.fit(np.array([[-0.7]]))