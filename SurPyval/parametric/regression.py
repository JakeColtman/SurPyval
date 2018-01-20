import pymc3 as pm

class SurvivalBlock(pm.Model):

    def __init__(self, log_lik_func, predictive_distribution):

        super(SurvivalBlock, self).__init__("", None)

        surv = pm.DensityDist('surv', log_lik_func, observed={'event': event, 'y': y})
        predictive = predictive_distribution()


class ExponentialSurvivalBlock(SurvivalBlock):

    def __init__(self):
        SurvivalBlock.__init__(self, self.log_lik, lambda: pm.Exponential("predictive", alpha, shape=y.shape))

    @staticmethod
    def log_lik(event, y):
        import theano.tensor as t
        return t.dot(event.T, t.log(alpha)) - t.dot(alpha.T, y)

    @staticmethod
    def parameters():
        return ["alpha"]


class RegressionBlock(pm.Model):
    def __init__(self, output_name):
        super(RegressionBlock, self).__init__("", None)
        import theano.tensor as t
        globals()[output_name] = pm.Deterministic(output_name, t.exp(t.dot(x, beta)))


def ExponentialRegression(prior_block: pm.Model) -> pm.Model:
    with prior_block as model:
        RegressionBlock("alpha")
        ExponentialSurvivalBlock()
    return model