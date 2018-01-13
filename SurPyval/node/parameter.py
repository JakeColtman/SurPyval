from typing import Dict, Any

from SurPyval.node.node import Node


class ParameterNode(Node):
    """
        For unobserved variables that need to be marginalized out

        Generally used to define prior distributions for parameters
    """

    def __init__(self, node: Node, name: str, length: int):
        self.name = name
        self.length = length
        self.distribution = node.distribution
        self.parameter_dict = node.parameter_dict
        self.constants_dict = node.constants_dict if node.constants_dict is not None else {}
        self.parameter_names = node.parameter_dict.keys()

    def sample(self, **kwargs):
        return kwargs[self.name]

    def __str__(self):
        return self.name
