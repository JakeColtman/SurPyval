
class KM:

    def __init__(self, times, event):
        '''
        Kaplan Meier estimator in its most basic form
        :param times: list of life times
        :param event: list of indicator variables where 1 indicates a death was seen
        '''
        self.data = self._preprocess_data(times, event)

    def _preprocess_data(self, times, event):
        return sorted(zip(times, event), key = lambda pair: pair[0])

    def fit(self):
        '''
        Fit the estimator using parameters already defined
        :return: self
        '''

        def period_survival_rate(number_alive):
            return 1 - (1 / number_alive)

        def calculate(data, past_risks):
            if len(data) == 0:
                return past_risks
            next_pair = data.pop(0)
            while next_pair[1] != 1:
                if len(data) == 0:
                    return past_risks
                next_pair = data.pop(0)

            period_rate = period_survival_rate(len(data) + 1)
            period_survival = past_risks[-1][1] * period_rate

            past_risks.append([next_pair[0], period_survival])

            return calculate(data, past_risks)

        return calculate(self.data, [[0, 1]])
