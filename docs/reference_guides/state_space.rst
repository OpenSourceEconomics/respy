The State Space
===============

.. currentmodule:: respy.state_space

The implementation of the state space in respy serves several purposes.
First and foremost the state space contains a register of all possible states of
the universe that a particular models allows for.
In high dimensional models the number of states grow substantially which constitutes
a considerable constraint for creating realistic models of economic dynamics.
To relax this constraint as much as possible it is crucial to avoid any
duplication of calculations or information.
To this extent we designed a state space that contains a range of different objects
that define representations and groupings of individual model states that allow us
to use as few resources as possible during the model solution.
Once an attribute of a state can be expressed as a combination of
the attributes of two sub-states the number of calculations required to
map all states to their attributes is reduced drastically.
The distinction between core and dense states within respy is defined such that
this property is exploited in a straight forward way.
Bundling states with a similar representation and a symmetric treatment in the
model solution together avoids double calculation and allows to write simpler
and more efficient code.
This consideration constitutes the base for the fine grained division of states
in respy which is defined in the period_choice_cores.
The range of methods contained in our state space facilitates clear communication
between different groups and representations of the state space.
This guide contains an explanation the most important components of the state space
in respy.

Representation
--------------
We divide state space dimensions in core and dense dimensions.
A core dimension is a deterministic function of past choices, time
and initial conditions.
Adding another core dimension changes the structure of the state space
in a complicated way.
A dense dimension is not such a function and thus changes the state
space in a more predictable way.
A prime example for a dense variable is an observable characteristic
that does not change in the model.
The addition of exogenous processes as dense variables makes the
distinction a bit less explicit but the general logic still applies.
A dense state of the model is a combination of its dense
and core states.
All attributes that are only functions of either dense dimensions or
core dimensions can be calculated seperately and can be combined
thereafter.

.. _core_state_space:

- The Core State Space:



.. _dense_grid:

- Dense Grid:
    The dense grid is a list that contains all states of dense variables. A dense
    variable is not a deterministic function of past choices and time. Adding another
    dense variables essentially copies the full state space. The disjunction makes
    use of this property. By expressing a state as a combination of a dense and a core
    state we avoid several duplications.

.. _dense_state_space:

- Dense State Space:
    The dense state space is the full state space that contains all admissible states.
    Respy does not store the full dense state space explicitly. The concept is
    nevertheless important since the model solution essentially loops through each
    dense state.


Division
--------

.. _period_choice_cores:

- Period Choice Cores:
    The interface of respy allows for flexible choice sets. The period choice core maps
    period and choice set to a set of core states.


.. _dense_period_choice_cores:

- Dense Period Choice Cores:
    The period choice core maps period and choice set and dense index to a set
    of core states. This is the separation of states that the model solution
    loops over. All of the state space operations work symmetrically on all states
    within one period. The same mapping is applied to generate the full set of
    covariates, they are all called at the same time during the backward induction
    procedure (reordering would not matter) and they all use the same rule to obtain
    continuation value. That is the reason why they are stored together and why the
    model solution loops over period and all ``dense_period_choice_cores`` within
    that period perform their operations in parallel.


.. _state_space_location_indices:

- State Space Location Indices:
    To store and manage information efficiently we build simple indices
    of the state space objects introduced above. In general these location
    indices are integers that point to a position within an object. In
    general we call location indices indices if the defining mapping is
    injective in the dense state space and keys if the defining mapping is
    not injective in the dense state space.
    Thus it follows that location indices that point to a row in the
    ``core_state_space`` are referred to as indices while location
    indices that refer to a group of states such as the numeration of the
    ``dense_period_choice_cores`` are referred to as keys.



Communication
-------------

.. _state_space_methods:

Several methods facilitate communication between different groups in the state space.
They are shortly introduced in turn:


.. _collect_child_indices:

- Collect Child Indices:
    This function assigns each state a function that maps choices into child states.


.. _get_continuation_values:

- Get Continuation Values:
    This method uses collect child indices to assign each state a function
    that maps choices into continuation values.
