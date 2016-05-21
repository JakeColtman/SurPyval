from unittest import TestCase
from lifelines.datasets import load_rossi
from SurPyval.CoxPH import CoxPH
from lifelines import CoxPHFitter

rossi_dataset = load_rossi()

class CoxPHTest(TestCase):

    def test_against_lifelines(self):

        rossi_dataset = load_rossi()
        cf = CoxPH(rossi_dataset, "week", "arrest")
        cf.fit()
        new_model = cf.hazards_
        cf = CoxPHFitter(normalize=False)
        cf.fit(rossi_dataset, 'week', event_col='arrest')
        old_model = cf.hazards_
        for estimate in zip(list(new_model), list(old_model.iloc[0])):
            self.assertEqual(estimate[0], estimate[1])



