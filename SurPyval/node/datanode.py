from SurPyval.node import Node


class DataNode(Node):
    """
    A Node for fixed sources of data in the model

    These nodes don't contribute to the likihood function at all
    Their only purpose is to store data for use in the model

    Parameters
    ----------
    name: str
          the name the data source has in the model (e.g. event, x)
    data: array-like
          the data itself, usually pd.DataFrame or np array

    """

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
