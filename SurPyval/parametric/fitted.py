import pymc3 as pm


class SurvivalBlock(pm.Model):

    def __init__(self, log_lik_func, predictive_distribution):

        super(SurvivalBlock, self).__init__("", None)

        y = pm.DensityDist('y', log_lik_func, observed={'event': data["event"], 'y': data["y"]})
        predictive = predictive_distribution()


class ExponentialSurvivalBlock(SurvivalBlock):

    def __init__(self):
        SurvivalBlock.__init__(self, self.log_lik, lambda: pm.Exponential("predictive", alpha))

    @staticmethod
    def log_lik(failure, value):
        import theano.tensor as t
        return (failure * t.log(alpha) - alpha * value).sum()

    @staticmethod
    def parameters():
        return ["alpha"]


def FittedExponential(prior_block: pm.Model) -> pm.Model:
    with prior_block as model:
        ExponentialSurvivalBlock()
    return model



