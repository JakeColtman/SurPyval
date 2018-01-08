import numpy as np

class NodeTree:

    def __init__(self, node_dict, parameters):
        self.parameters = parameters
        self.node_dict = node_dict
        self.node_names = sorted(node_dict.keys())
        self.flat_split_point = self.flattened_parameter_split_points()

    def log_lik(self, flattened_parameters):
        unflattened_parameters = self.unflatten_parameter_array(flattened_parameters)
        return np.sum(map(lambda x: x.log_lik(**unflattened_parameters), self.node_dict.values()))

    def add_node(self, node_name, node):
        new_dict = self.node_dict
        new_dict[node_name] = node
        return NodeTree(new_dict, self.parameters)

    def flattened_parameter_split_points(self):
        split_points = [0]
        for parameter in self.parameters:
            split_points.append(split_points[-1] + parameter.length)
        return split_points[1:]

    def length(self):
        return np.sum([x.length for x in self.parameters])
    
    def unflatten_parameter_array(self, flat_parameter_array):
        split_array = np.split(flat_parameter_array, self.flat_split_point)
        return {x[0]: x[1] for x in zip([x.name for x in self.parameters], split_array)}
    
    def flatten_parameter_dict(self, parameter_dict):
        flat_list = []
        for node_name in self.node_names:
            flat_list.append(parameter_dict[node_name])
        return np.array(flat_list)

if __name__ == "__main__":
    import unittest

    class TestNodeTree(unittest.TestCase):

        def test_add_node(self):
            node_dict = {
                "beta": exponential(1.0, ["beta"]),
                "gamma": exponential(2.0, ["gamma"])
            }
            parameters = [Parameter("beta", 1.0), Parameter("gamma", 1.0)]
            tree = NodeTree(node_dict, parameters)
            new_tree = tree.add_node("gaussian", gaussian(np.array([1.0, 2.0]), np.diag([2.0, 2.0]), ["x"]))
            self.assertEqual(len(new_tree.node_dict), 3)
            self.assertTrue("gaussian" in new_tree.node_dict)
        
        def test_two_nodes_sharing_a_variable(self):
            node_dict = {
                "beta_0": exponential(1.0, ["beta"]),
                "beta_1": exponential(2.0, ["beta"]),
                "gamma": exponential(3.0, ["gamma"])
            }

            parameters = [Parameter("beta", 1.0), Parameter("gamma", 1.0)]
            tree = NodeTree(node_dict, parameters)
            
            unflattened_parameters = tree.unflatten_parameter_array(np.array([8.0, 2.0]))
            
            self.assertEqual(unflattened_parameters["beta"][0], 8.0)
            self.assertEqual(unflattened_parameters["gamma"][0], 2.0)
            
        
        def test_basic_parameter_unflattening(self):
            node_dict = {
                "beta": exponential(1.0, ["beta"]),
                "gamma": exponential(2.0, ["gamma"])
            }

            parameters = [Parameter("beta", 1.0), Parameter("gamma", 1.0)]
            tree = NodeTree(node_dict, parameters)
            
            unflattened_parameters = tree.unflatten_parameter_array(np.array([8.0, 5.0]))
            
            self.assertEqual(unflattened_parameters["beta"][0], 8.0)
            self.assertEqual(unflattened_parameters["gamma"][0], 5.0)
            
        def test_basic_parameter_flattening(self):
            node_dict = {
                "beta": exponential(1.0, ["beta"]),
                "gamma": exponential(2.0, ["gamma"])
            }

            parameters = [Parameter("beta", 1.0), Parameter("gamma", 1.0)]
            tree = NodeTree(node_dict, parameters)
            
            flattened_parameters = tree.flatten_parameter_dict({"gamma": 5.0, "beta": 8.0})
            
            self.assertEqual(flattened_parameters[0], 8.0)
            self.assertEqual(flattened_parameters[1], 5.0)
    
            
        def test_flattening_inverse_of_unflattening(self):
            node_dict = {
                "beta": exponential(1.0, ["beta"]),
                "gamma": exponential(2.0, ["gamma"])
            }

            parameters = [Parameter("beta", 1.0), Parameter("gamma", 1.0)]
            tree = NodeTree(node_dict, parameters)
            
            original_unflattened_parameters = {"gamma": 5.0, "beta": 8.0}
            flattened_parameters = tree.flatten_parameter_dict(original_unflattened_parameters)
            unflattened_parameters = tree.unflatten_parameter_array(flattened_parameters)
            
            for key in original_unflattened_parameters:
                self.assertEqual(unflattened_parameters[key], original_unflattened_parameters[key])

    suite = unittest.TestLoader().loadTestsFromTestCase(TestNodeTree)
    unittest.TextTestRunner(verbosity=2).run(suite)