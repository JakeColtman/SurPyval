# WIP - API subject to sweeping changes

# SurPyval

SurPyval is a Bayesian survival analysis library

## Philosophy:
    
    * Models should be transparent about their assumptions and workings
    * Models should allow tweaks and modifications

Implementing this philosophy has a number of positive effects on the library:

    * The log-likihood and plate diagrams of models are exposed
    * Models are created through composition of simple units
    * SurPyval objects thinly wrap and expose well-know libraries (esp. scipy)
    * There are no hand-offs to non-python objects
    * Models allow for substitution of any of their composite blocks
    
In general, the trade-off to buy the above is speed and memory performance.  Constructing models in a compositional, modifiable way often leads to not being able to get performance improvements.  For medium sized data sets, this isn't a problem, but for very large data sets and complex models speed can be a problem.