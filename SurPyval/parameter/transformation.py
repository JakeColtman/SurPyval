class Transformation(object):
    """
        Defines a transformation from (data, parameter) -> value
        Using transformed versions of variables simplifies down stream code

        Example use case:

            Transforming beta into the linear predictor in exponential regression
    """

    def __init__(self, f, new_name):
        self.f = f
        self.new_name = new_name

    def transform(self, data_dict, parameter_dict):
        return {self.new_name: self.f(data_dict, parameter_dict)}