# WIP - API subject to sweeping changes

# SurPyval

SurPyval is a Bayesian survival analysis library.

## Why use

SurPyval is aimed at people who are using survival analysis libraries like lifelines, but who want more flexibility and access to Bayesian approaches.

Users who are new to Bayesianism can make use of sensible defaults and helper methods, while power users can take extremely detailed control of models.

## Philosophy:

SurPyval is built on the core philosophy that it should be as easy as possible for users to understand and tweak models.  Many statistical libraries are really easy to use until you want a slightly different assumption, which they cannot support.

There are two main architectural decisions this has entailed:

#### Graph centric

All models in SurPyval are build around graphical models.  Every model can return a plate diagram of its likelihood function.  Digger deeper, models are actually made through composing together different variable "Nodes".  To change the model, we can simply swap out or add nodes to the model's graph

#### Thin wrapper over common libraries

Allowing modification and tweaking is much less valuable if doing so requires learning a complex new API.  To make the process as simple as possible, most SuPyval classes and objects are relatively thin wrappers over classes from libraries like scipy and emcee.  SurPyval objects are eager to expose these common libraries to the user

#### Trade offs

In general, SuPyval is comfortable with paying for composability and modifiability with performance.  For a lot of tasks, SurPyval won't run as quickly as (say) a custom written Stan model.  However:
    
    * For data sets with 6 figure rows, the slow down isn't much of a problem
    * Any performance loss is asymetrrical, some use cases will be blazing fast
    * There are ways of mitigating this (see performance part of docs)