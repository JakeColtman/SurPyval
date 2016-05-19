from lifelines import KaplanMeierFitter

class KM:

    def __init__(self, times, event):
        '''
        Kaplan Meier estimator in its most basic form
        :param times: list of life times
        :param event: list of indicator variables where 1 indicates a death was seen
        '''
        self.data = self._preprocess_data(times, event)

    def _preprocess_data(self, times, event):
        '''
        Sort two lists into a single list of lists ordered by time ascending and with events before censors
        (makes life easier when censors and events happen at the same time
        '''
        return sorted(zip(times, event), key = lambda pair: (pair[0], pair[1] * - 1))

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

            if past_risks[-1][0] == next_pair[0]:
                past_risks[-1][1] = period_survival
            else:
                past_risks.append([next_pair[0], period_survival])

            return calculate(data, past_risks)

        return calculate(self.data, [[0, 1]])

if __name__ == "__main__":
    times = [1, 2, 2, 3]
    event = [1, 0, 1, 1]
    km = KM(times, event)
    print(km.fit())
    fitter = KaplanMeierFitter()
    lifelines_estimate = sorted(list(set(fitter.fit(times, event).survival_function_["KM_estimate"])), reverse = True)
    print(lifelines_estimate)