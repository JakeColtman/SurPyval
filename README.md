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
    
The trade-off to get these goods is performance.  Models provide in the library are designed to be tweakable, which limits performance optimizations.  This manifests itself in a number of ways:

    * Straight up crunching speed
    * Memory useage
    * Models often don't exploit conjugacy where it exists
    
For very large data sets or very complicated models, you might be better off using something like Stan.