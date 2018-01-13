

class Parameter:
    """
        Wrapper for Parameters in the model
        Holds the name of a parameter for routing and its length (number of elements)

        Example use:
            * If fitting an exponential distribution to data, `alpha` would be a parameter with 1 element so:
                `Parameter("alpha", 1)`

        Names of Parameters can be set at will, they don't need to match up with the naming in distributions
        e.g. we could instead have done `Parameter("ImADifferentNameForAlpha", 1)
    """

    def __init__(self, name, length):
        self.name = name
        self.length = length

    def __str__(self):
        return self.name
