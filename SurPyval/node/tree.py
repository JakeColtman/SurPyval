import numpy as np
from typing import Dict, List, Any

from SurPyval.node.parameter import ParameterNode
from SurPyval.node.node import Node
from SurPyval.node.transformation import DeterministicNode
from SurPyval.node.datalikihoodnode import DataLikihoodNode


class NodeTree:
    """
        Encapsulates the graphical model and allows clients to interact with the log-likihood

        Parameters
        ----------
        node_dict: Dict[str, Node]
                   lookup from name of node a Node
        data_dict: Dict[str,array-like]
                   lookup from name to data (e.g. (event->np.array([True, False, True])
    """

    def __init__(self, node_dict: Dict[str, Node], data_dict: Dict[str, Any]):
        self.parameters: List[ParameterNode] = [x for x in node_dict.values() if type(x) is ParameterNode]
        self.transformations: List[DeterministicNode] = [x for x in node_dict.values() if type(x) is DeterministicNode]
        self.likihood_nodes: List[DataLikihoodNode] = [x for x in node_dict.values() if type(x) is DataLikihoodNode]

        self.data_dict = data_dict
        self.node_dict = node_dict
        self.node_names = sorted(node_dict.keys())
        self.flat_split_point = self.flattened_parameter_split_points()

    def append_transformations(self, parameter_dict):
        """
        Enrich the parameter dictionary with transformed parameters
        Transformed variables will be indistinguishable from original parameters

        Parameters
        ----------
        parameter_dict: Map<parameter_name: str, value: np.array>

        Returns
        -------
        Map<parameter_name: str, value: np.array>
        """
        parameter_dicts_from_transformations = [
            x.transform(self.data_dict, parameter_dict) for x in self.transformations
        ]
        for parameter_dict_from_transformations in parameter_dicts_from_transformations:
            parameter_dict.update(parameter_dict_from_transformations)

        return parameter_dict

    def logpdf(self, flattened_parameters):
        """
        Access point for other classes to apply numerical methods to loglikihood
        Allows classes to pass in a flat np.array and have the result be parsed

        Parameters
        ----------
        flattened_parameters: np.array[float] with length self.length

        Returns
        -------
        float
        """
        unflattened_parameters = self.unflatten_parameter_array(flattened_parameters)
        dict_to_pass = {**unflattened_parameters, **self.data_dict}
        return np.sum([x.logpdf(**dict_to_pass) for x in self.node_dict.values()])

    def generate_replicate(self, flattened_parameters):
        """
        Draw a realization the model conditional on the value of parameters

        Parameters
        ----------
        flattened_parameters: array-like
                              length should be equal self.length

        Returns
        -------
        Dict[str, array-like]
            lookup from node to samples
        """
        unflattened_parameters = self.unflatten_parameter_array(flattened_parameters)
        return {x: x.sample(**unflattened_parameters) for x in self.likihood_nodes}

    def flattened_parameter_split_points(self):
        """
        Calculate the correct split points in the flat array to assign each number to the correct parameter

        Returns
        -------
        List[Int] - length = len(self.parameters)
        """
        split_points = [0]
        for parameter in self.parameters:
            split_points.append(split_points[-1] + parameter.length)
        return split_points[1:]

    def length(self):
        """
        Total width of the parameter array, i.e. the number of estimated parameters

        Note: this length doesn't include the deterministic transformed parameters

        Returns
        -------
        int
        """
        return int(np.sum([x.length for x in self.parameters]))
    
    def unflatten_parameter_array(self, flat_parameter_array):
        split_array = np.split(flat_parameter_array, self.flat_split_point)
        param_dict = {x[0]: x[1] for x in zip([x.name for x in self.parameters], split_array)}
        return self.append_transformations(param_dict)
    
    def flatten_parameter_dict(self, parameter_dict):
        flat_list = []
        for node_name in self.node_names:
            flat_list.append(parameter_dict[node_name])
        return np.array(flat_list)

#import unittest
#
# class TestNodeTree(unittest.TestCase):
#
#     def test_add_node(self):
#         node_dict = {
#             "beta": exponential(1.0, {"beta": "alpha"}),
#             "gamma": exponential(2.0, {"gamma": "alpha"})
#         }
#         parameters = [Parameter("beta", 1.0), Parameter("gamma", 1.0)]
#         tree = NodeTree(node_dict, {}, parameters, [])
#         new_tree = tree.add_node("gaussian", gaussian(np.array([1.0, 2.0]), np.diag([2.0, 2.0]), {"beta": "x"}))
#         self.assertEqual(len(new_tree.node_dict), 3)
#         self.assertTrue("gaussian" in new_tree.node_dict)
#
#     def test_two_nodes_sharing_a_variable(self):
#         node_dict = {
#             "beta_0": exponential(1.0, {"beta": "alpha"}),
#             "beta_1": exponential(2.0, {"beta": "alpha"}),
#             "gamma": exponential(3.0, {"gamma": "alpha"})
#         }
#
#         parameters = [Parameter("beta", 1.0), Parameter("gamma", 1.0)]
#         tree = NodeTree(node_dict, {}, parameters, [])
#
#         unflattened_parameters = tree.unflatten_parameter_array(np.array([8.0, 2.0]))
#
#         self.assertEqual(unflattened_parameters["beta"][0], 8.0)
#         self.assertEqual(unflattened_parameters["gamma"][0], 2.0)
#
#
#     def test_basic_parameter_unflattening(self):
#         node_dict = {
#             "beta": exponential(2.0, {"beta": "alpha"}),
#             "gamma": exponential(3.0, {"gamma": "alpha"})
#         }
#
#         parameters = [Parameter("beta", 1.0), Parameter("gamma", 1.0)]
#         tree = NodeTree(node_dict, {}, parameters, [])
#
#         unflattened_parameters = tree.unflatten_parameter_array(np.array([8.0, 5.0]))
#
#         self.assertEqual(unflattened_parameters["beta"][0], 8.0)
#         self.assertEqual(unflattened_parameters["gamma"][0], 5.0)
#
#     def test_basic_parameter_flattening(self):
#         node_dict = {
#             "beta": exponential(1.0, {"beta": "alpha"}),
#             "gamma": exponential(2.0, {"gamma": "alpha"})
#         }
#
#         parameters = [Parameter("beta", 1.0), Parameter("gamma", 1.0)]
#         tree = NodeTree(node_dict, {}, parameters, [])
#
#         flattened_parameters = tree.flatten_parameter_dict({"gamma": 5.0, "beta": 8.0})
#
#         self.assertEqual(flattened_parameters[0], 8.0)
#         self.assertEqual(flattened_parameters[1], 5.0)
#
#
#     def test_flattening_inverse_of_unflattening(self):
#         node_dict = {
#             "beta": exponential(2.0, {"beta": "alpha"}),
#             "gamma": exponential(3.0, {"gamma": "alpha"})
#         }
#
#         parameters = [Parameter("beta", 1.0), Parameter("gamma", 1.0)]
#         tree = NodeTree(node_dict, {}, parameters, [])
#
#         original_unflattened_parameters = {"gamma": 5.0, "beta": 8.0}
#         flattened_parameters = tree.flatten_parameter_dict(original_unflattened_parameters)
#         unflattened_parameters = tree.unflatten_parameter_array(flattened_parameters)
#
#         for key in original_unflattened_parameters:
#             self.assertEqual(unflattened_parameters[key], original_unflattened_parameters[key])
