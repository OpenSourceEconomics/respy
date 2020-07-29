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
This consideration cosnitutes the base for the fine grained division of states
in respy which is defined in the period_choice_cores.
The range of methods contained in our state space facilitates clear communication
between different groups and representations of the state space.
This guide contains an explanation the most important components of the state space
in respy.


.. _core_state_space:

The Core State Space
--------------------

ads

.. _dense_grid:

Dense Grid
----------

The dense grid is a list that contains all states of dense variables. A dense variable
is not a deterministic function of past choices and time. Adding another dense variables
essentially copies the full state space. The disjunction makes use of this property. By
expressing a state as a combination of a dense and a core state we avoid several
duplications.


.. _dense_state_space:

Dense State Space
-----------------

The dense state space is the full state space that contains all admissible states. respy
does not store the full dense state space explicitly. The concept is nevertheless
important since the model solution essentially loops through each dense state.


.. _period_choice_cores:

Period Choice Cores
-------------------

The interface of respy allows for flexible choice sets. The period choice core maps
period and choice set to a set of core states.


.. _dense_period_choice_cores:

Dense Period Choice Cores
-------------------------

The period choice core maps period and choice set and dense index to a set of core
states. This is the separation of states that the model solution loops over. All of the
state space operations work symmetrically on all states within one period. The same
mapping is applied to generate the full set of covariates, they are all called at the
same time during the backward induction procedure (reordering would not matter) and they
all use the same rule to obtain continuation value. That is the reason why they are
stored together and why the model solution loops over period and all
``dense_period_choice_cores`` within that period perform their operations in parallel.


.. _state_space_location_indices:

State Space Location Indices
============================

To create the state space and to store information efficiently we build simple indices
of the objects introduced above. In general we call location indices indices if the
defining mapping is injective and keys if the defining mapping is not injective.


.. _core_indices:

Core Indices
------------

Core indices are row indices for states in the :ref:`core state space
<core_state_space>`. They are continued over different periods and choice sets in the
core.


.. _core_key:

Core Key
--------

A ``core_key`` is an index for a set of states in the core state space which are in the
same period and share the same choice set.


.. _dense_vector:

Dense Vector
------------

A dense vector is combination of values in the dense dimensions.


.. _dense_index:

Dense Index
-----------

A dense index is a position in the dense grid.


.. _dense_key:

Dense Key
---------

A ``dense_key`` is an index for a set of states in the dense state space which are in
the same period, share the same choice set, and the same dense vector.

.. _complex:

Complex
--------------------
A complex key is the basis for `core_key` and `dense_key` it is a tuple of a period and
a tuple for the choice set which contains booleans for whether a choice is available.
The complex index for a dense index also contains the dense vector in the last position.

.. _state_space_methods:


State Space Methods
===================

Several methods facilitate communication between different groups in the state space.
They are shortly introduced in turn:


.. _collect_child_indices:

Collect Child Indices
---------------------

This function assigns each state a function that maps choices into child states.


.. _get_continuation_values:

Get Continuation Values
-----------------------

This method uses collect child indices to assign each state a function
that maps choices into continuation values.
