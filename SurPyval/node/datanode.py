from SurPyval.node.node import Node


class DataNode(Node):

    def __init__(self, name, data):
        self.name = name
        self.data = data

    def sample(self, size=1, **kwargs):
        return None

    def logpdf(self, **kwargs):
        return 0.0

    def pdf(self, **kwargs):
        return 1.0

    def __str__(self):
        return self.name
