import pymc3 as pm


class FixedSplitPointsBlock(pm.Model):

    def __init__(self, split_point_times):
        super(FixedSplitPointsBlock, self).__init__("", None)
        import theano.tensor as t
        internal_split_points = t._shared(split_point_times)
        globals()["split_points"] = pm.Deterministic("split_points", t.concatenate([_shared(np.array([0])), internal_split_points, _shared(np.array([y.max()]))]))
        globals()["period_lengths"] = pm.Deterministic("period_lengths", (t.roll(split_points, -1) - split_points)[:-1])


class IndepGammaConstantHazardsBlock(pm.Model):

    def __init__(self, alpha_0, llambda_0, shape):
        super(IndepGammaConstantHazardsBlock, self).__init__("", None)
        globals()["llambda"] = pm.Gamma("llambda", alpha_0, llambda_0, shape=shape)


class ProcessedBaselineHazards(pm.Model):

    def __init__(self):
        super(ProcessedBaselineHazards, self).__init__("", None)
        globals()["period_of_death"] = t.extra_ops.searchsorted(split_points, y) - 1

        globals()["start_time_of_death_period"] = split_points[period_of_death]
        globals()["constant_hazard_of_death_period"] = llambda[period_of_death]
        globals()["time_in_death_period"] = pm.Deterministic("time_in_death_period", (y_shared - start_time_of_death_period))

        globals()["total_baseline_hazard_in_death_period"] = constant_hazard_of_death_period * time_in_death_period

        globals()["did_complete_period"] = t.ge(t.outer(y_shared, t.ones_like(split_points[1:])), split_points[1:])
        globals()["total_baseline_exposure_in_finished_periods"] = (did_complete_period * llambda * period_lengths).sum(axis=1)

        globals()["total_baseline_hazard_exposure"] = total_baseline_exposure_in_finished_periods + total_baseline_hazard_in_death_period


class RegressionBlock(pm.Model):
    def __init__(self, output_name):
        super(RegressionBlock, self).__init__("", None)
        import theano.tensor as t
        globals()[output_name] = pm.Deterministic(output_name, t.exp(t.dot(x, beta)))


class ExponentialLikihoodBlock(pm.Model):

    def __init__(self):
        super(ExponentialLikihoodBlock, self).__init__("", None)
        globals()["surv"] = pm.DensityDist('surv', self.log_lik, observed={'event': event, 'y': y})
        globals()["predictive"] = pm.Exponential("predictive", )

    @staticmethod
    def log_lik(event, y):
        import theano.tensor as t
        lhs = t.log(t.dot(constant_hazard_of_death_period * event, eta))
        rhs = t.dot(total_baseline_hazard_exposure, eta)
        return lhs + rhs

