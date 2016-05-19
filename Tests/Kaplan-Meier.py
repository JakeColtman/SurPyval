from lifelines import KaplanMeierFitter
from matplotlib.pyplot import show
from SurPyval.KM import KM
import unittest
import pandas as pd

class KaplanMeierTestsAgainstLifelines(unittest.TestCase):

    def test_with_no_duplicates(self):
        times = [2, 4, 6, 3, 5, 7, 9, 12, 11, 10, 1]
        event = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
        fitter = KaplanMeierFitter()
        lifelines_estimate = sorted(list(set(fitter.fit(times, event).survival_function_["KM_estimate"])), reverse = True)
        surPyval_estimate = [x[1] for x in KM(times, event).fit()]
        for pair in zip(lifelines_estimate, surPyval_estimate):
            self.assertAlmostEqual(pair[0], pair[1], 2)