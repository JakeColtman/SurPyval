import numpy as np

from SurPyval.model import Model
from SurPyval.node import DeterministicNode, DataNode
from SurPyval.semiparametric.piecewise.datalikihoodnode import DataLikihood

class ProportionalHazards(Model):

    def __init__(self, y, event, x, break_points, llambda_prior, beta_prior):

        node_dict = {
            "llambda": llambda_prior,
            "beta": beta_prior,
            "s": DataNode("s", break_points),
            "x": DataNode("x", x),
            "event": DataNode("event", event),
            "eta": DeterministicNode(lambda data_dict, param_dict: np.exp(np.dot(data_dict["x"], param_dict["beta"])), "eta")
            "hazard_exposure": DeterministicNode(la),
            "hazard_event_prior": DeterministicNode()
        }