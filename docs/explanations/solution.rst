.. _solution_model:

Mathematical Framework
======================

.. role:: boldblue

Mathematically, the class of EKW models (so the flagship model in Keane and Wolpin
(1997, :cite:`Keane.1997`)) is set up as a standard
:boldblue:`Markov decision process` (Puterman, 1994, :cite:`Puterman.1994`;
White, 1993, :cite:`White.1993`). Individuals make sequential decisions
under uncertainty and aim to implement an optimal policy :math:`\pi^* \in \Pi`,
given the states of the world.

In principle, this requires to evaluate the performance of all policies
:math:`\Pi` based on all possible sequences of per-period utilities, each of
them weighted by the probability with which they occur. Fortunately, the
multistage problem can be solved by a sequence of simpler inductively defined
single-stage problems. Puterman (1994, Chapter 4, :cite:`Puterman.1994`)
shows that deterministic Markovian decision rules are always optimal in our
presented economic setting. Hence, we will restrict our exposition to this
special case.

For any given state :math:`s_t`, the value function :math:`v_t^{\pi}(s_t)`
captures the expected total discounted per-period utilities under policy
:math:`\pi` from period :math:`t` onwards

.. math::

   v_t^{\pi}(s_t) \equiv \mathbb{E}_{p^{\pi}} \left[ \sum_{j=0}^T \delta^j
   u_{t+j}(s_{t+j}), a_{t+j}^{\pi}(s_{t+j}) \,\big|\, s_t \right]

For any policy :math:`\pi` we can determine the value function at the initial
state by recursively evaluating

.. math::
    :label: eq:recursive_evaluation

    v_t^{\pi}(s_t) = u(s_t, a_t^{\pi}(s_t)) + \delta \,\mathbb{E}_{p^{\pi}}
    \left[ v_{t+1}^{\pi}(s_{t+1}) \,|\, s_t \right].

Invoking the principle of optimality (Bellman, 1954, 1957, :cite:`Bellman.1954`,
:cite:`Bellman.1957`)) allows to construct the optimal policy :math:`\pi^*`
by solving the optimality equations for all :math:`s_t` recursively

.. math::
    :label: eq:bellman_optimality_equations

    v_t^{\pi^*}(s_t) = \underset{a_t \in \mathcal{A}}{\max}
    \left\{ u(s_t, a_t^{\pi}(s_t)) + \delta \,\mathbb{E}_{p^{\pi}}
    \left[ v_{t+1}^{\pi^*}(s_{t+1}) \,|\, s_t \right] \right\}.

The optimal value function :math:`v_t^{\pi^*}` is the sum of expected discounted
utility in :math:`t` over remaining assumption given that the optimal policy is
implemented in all subsequent periods.

--------------------------------------------------------------------------------

.. rst-class:: centerblue

  In `respy` the value functions can be retrieved as part of any model solution.
  In the case of our Keane and Wolpin (1997) exposition the value functions are
  indexed via ``Value_Function_{Alternative}``.

--------------------------------------------------------------------------------

The :ref:`algorithm_value_function_iteration` solves the Markov decision process
via backward induction procedure and retrieves the decision rules of the agents.
In the final period :math:`T` there is no future to take into account, and the
optimal action is choosing the alternative with the highest per-period utilities
in each states. The preceding decision rules are determined recursively.

.. _algorithm_value_function_iteration:

.. figure:: ../_static/images/algorithm_value_function_iteration.pdf
  :width: 650
  :alt: Algorithm value function iteration

  Value Function Iteration Algorithm


Solving the Integrated Value Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As already suggested, the state space contains a stochastic component
:math:`\epsilon_t`. Equation (2) constitutes the major reason for the
computational complexity of DCDP. The integrated value function

.. math::
  :label: eq:emax

  \text{Emax}(s_{t}) &\equiv \int_{\mathcal{S}}
  v_{t+1}^{\pi^*}(s_{t+1}) \, \mathrm{d}p(s_t, a_t) \\
  &= \int_{\mathcal{S}} \underset{a \in \mathcal{A}}{\max}
  \left\{ v_{t+1}^{\pi}(s_{t+1}) \right\} \, \mathrm{d}p(s_t, a_t)

has no analytical solution, and so numerical methods have to be applied.
Keane and Wolpin impose two assumptions in order to provide a simplification of
the expression:

  - Stochastic shocks :math:`\epsilon_{t}` are independently and identically
    distributed over individuals and time (serially uncorrelated), conditional on
    :math:`s_{t}`. We will denote their probability density function
    :math:`\phi_{\mu, \Sigma}(\epsilon_{t})`.

  - State variables are independent of the realizations of :math:`\epsilon_{t}`,
    conditional on decisions. This is the reason we can write
    :math:`p(s_t, \epsilon_t, a_t) = p(s_t, a_t)`.

We can reformulate parts of the integrated value function

.. math::
  :label: eq:emax_reformulated

  \int_{\mathcal{S}} \, \underset{a \in \mathcal{A}}{\max} \,
  \left\{  v_{t+1}^{\pi}(s_{t+1}) \right\} \mathrm{d}p(s_t, a_t)
  = \int_{\epsilon} \underset{ a \in \mathcal{A}}{\max} \,
  \left\{  v_{t+1}^{\pi}(s_{t+1}) \right\} \,
  \mathrm{d}\phi_{\mu, \Sigma}(\epsilon_{t}).

This expression is a :math:`(|\mathcal{A} | = 5)`-dimensional integral
which has to be solved for any possible state :math:`s_{t} \in \mathcal{S}_t`,
hence million-wise. [#]_

Most of the current implementations use Monte Carlo integration to solve the
integral numerically. Judd and Skrainka (2011, :cite:`Judd.2011`) lament the
resulting numerical errors and computational instabilities.

--------------------------------------------------------------------------------

.. rst-class:: centerblue

   The EMax calculation in ``respy`` relies on advanced methods. The use of
   quasi Monte-Carlo methods mitigates numerical errors and dramatically reduces
   the time to solve the model.

   A How-to guide is provided in `Improving Numerical Integration Methods
   <https://respy.readthedocs.io/en/latest/how_to_guides/numerical_
   integration.html>`_.

--------------------------------------------------------------------------------

The formulation in Equation :eq:`emax_reformulated` indicates that the
:boldblue:`computational complexity` is governed by the size of the
(observable) state space [#]_ and the multi-dimensionality of the
integral. Notably, to retrieve the optimal policy :math:`\pi^*` it is
necessary to solve the value function at each point of the state space.
This demonstrates the so-called  ''curse of dimensionality'' (Bellman, 1957,
:cite:`Bellman.1957`). The number of states grows exponentially with the number
of available choices (:math:`|\mathcal{A}|`) and linearly in the number of
periods. Every possible extension of the Keane and Wolpin (1997) model that
affects any of both factors will be computationally more demanding.


A comparison of Keane and Wolpin (1997, :cite:`Keane.1997`) and
Keane and Wolpin (2000, :cite:`Keane.2000`) quantifies this link between state
space and computational complexity. In Keane and Wolpin (2000, :cite:`Keane.2000`)
the authors enrich the model with a dummy variable that captures a binary
characteristic of the individual decision-maker. This binary state option
increases the state space from initially 52 million states to 104 million states
in Keane and Wolpin (2000, :cite:`Keane.2000`) . For a given parameterization of
the model it is necessary to evaluate Equation :eq:`EMax` at each of the points.


.. rubric:: Footnotes

.. [#] This becomes particularly bothersome in estimation where many trial
       parameter values are tested. Then it is necessary to evaluate
       Equation :eq:`EMax` for any trial parametrization at all state points.

.. [#] The state space is given by :math:`s_t = (\bar{s}_t, \epsilon_t)`,
       where :math:`\bar{s}_t` constitutes the observable part.
