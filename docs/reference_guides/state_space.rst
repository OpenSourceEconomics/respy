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
This guide contains an illustration of the most important components of the state space
by stressing their role in representation of states, division of states and
communication between states.


Representation
--------------
We divide state space dimensions in core and dense dimensions.
A core dimension is a deterministic function of past choices, time
and initial conditions.
Adding another core dimension changes the structure of the state space
in a complicated way.
A dense dimension is not such a function and thus either changes the state space
in a more predictable way or changes the state space in such a complicated way that
we find it easier to have slightly larger state space than required rather than
performing complciated calculations to get a precise one.
A prime example for a dense variable in pur mode is an observable characteristic
that does not change in the model.
The addition of exogenous processes as dense variables makes the
distinction a bit less explicit but the general logic still applies.
A dense state of the model is a combination of its dense
and core states.
All attributes that are only functions of either dense dimensions or
core dimensions can be calculated seperately and can be combined
thereafter.
This property dramatically reduces the amount of calculations required
to obtain several pieces of information.
The main conceptual objects underlying the representation of states
are the core state space, the dense grid and the dense state space.

.. _core_state_space:

- The Core State Space:
    The core state space is the set of all core states. It is the image of the function
    underlying the core dimensions invoked at al feasible combinations of past choices
    time and initial conditions.
    The Core State Space is explicitely obtained in the module and is represented by
    a pd.DataFrame. As such it is the basis for most calculations and every state in the
    solution is partly represented by apointer to a row in the core state space.



.. _dense_grid:

- Dense Grid:
    The dense grid is a list that contains all states of dense variables.
    The main difference to the core state space is that we do not apply logic to obtain
    all feasible states of dense dimensions.
    Instead we just use the full cartesian product of dense dimenions. As we already
    mentioned in the last section this is either due to the fact that the marginal change
    of the state space due to the addition of the variable is so simple that we do not
    have to do any more than to duplicate the existing state space or due to the fact
    the marginal change is so complicated that we are fine with having a slightly bigger
    state space than required.
    The dense grid is also explicitely obtained and represented by a list of tuples.
    Each state in the solution is partly represented by a pointer to a position in the
    dense grid.

.. _dense_state_space:

- Dense State Space:
    The dense state space contains all full states that the model allows for.
    In respy the dense state space is essentially the cartesian product of the core
    state space and the entries in the dense grid.
    We however do not store the full dense state space explicitly.
    We rather create the dense state space sperately for each
    dense_period_choice_chore.
    Each dense state is represented by a combination of the dense state
    and the subset of the core dataframe that corresponds to the particular
    denste_period_choice_core. We only require the full information at two
    particular points in the creation of the state space and the solution
    of the model. Since it takes a long time to create this object and since it
    would consume a lot of memory to keep it in the working memory at all times
    the objects are saved to disk after they are created and only called whenever
    they are required.

Division
--------
The essential dimension along which our model is solved is time. That implies
that the minimal division of the dense state space that we require to solve our model
is along time.
During the solution and analysis different states however are treated differently. In
particular covariates are calculated differently and choices or other conditions are
different.
We thus want a division of the state space in each period that is as symmetric as
possible in its treatment during the solution while not being too complicated to
compile and manage.
The dense state space is seperated in ``dense_period_choice_cores`` in respy and several
objects and indices are created along the way.

.. _period_choice_cores:

- Period Choice Cores:
    The interface of respy allows for flexible choice sets.
    The period choice core maps period and choice set to a set of core states.
    It mainly constitutes the base for dense_period_choice cores.


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
    of the state space objects introduced above.
    It is crucial to say that we only need certain information at certain
    points in the solution process. Thus we define a location indices such that we
    can access all the information that we need throughout the solution easily.
    In general these location indices are integers that point to a position
    within an object. In general we call location indices indices if the defining
    mapping is injective in the dense state space and keys if the defining mapping is
    not injective in the dense state space. Thus it follows that location indices that
    point to a row in the ``core_state_space`` are referred to as indices while location
    indices that refer to a group of states such as the numeration of the
    ``dense_period_choice_cores`` are referred to as keys.



Communication
-------------

.. _state_space_methods:

Several methods facilitate communication between different groups in the state space.
To solve a discrete dynamic choice model dense period choice cores in different
periods need to communicate each other.
This section shortly summarizes how the three main state space methods create an
intertemporal link between state space groups and thereby facilitate an efficient model
solution. 

.. _collect_child_indices:

- Collect Child Indices:
    This function assigns each state a function that maps choices into child states.
    (Requires more information)

.. _get_continuation_values:

- Get Continuation Values:
    This method uses collect child indices to assign each state a function
    that maps choices into continuation values.
    It is the api that a dense period choice core uses to get information about the
    continuation values.

.. _set_attribute_from_keys:

- Set Attribute from keys:
    This method allows to store information at a certain point of a state space object.
    It is the api that a dense period choice core uses to store information about the
    expected value functions.
