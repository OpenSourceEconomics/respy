.. _economic_model:

Economic Model
==============

.. role:: boldblue

To illustrate the usage of **respy** in solving :boldblue:`finite-horizon
discrete choice dynamic programming` (DCDP) problems we will present the basic
setup of the human capital model as conceptualized in Keane and Wolpin
(1997, :cite:`Keane.1997`).

--------------------------------------------------------------------------------

.. rst-class:: centerblue

   The promise: **respy** will become your preferred tool
   to develop, solve, estimate, and explore models within the EKW framework

--------------------------------------------------------------------------------

At time t = 1,...,T each individual observes the state of the economic environment
:math:`s_{t} \in \mathcal{S}_t` and chooses an action :math:`a_t` from the set of
admissible actions :math:`\mathcal{A}`. The decision has two consequences:

  - Individuals receive per-period utility :math:`u_a(\cdot)` associated with the
    chosen alternative :math:`a_t \in \mathcal{A}`. [#]_
  - The economy evolves from :math:`s_{t}` to :math:`s_{t+1}`.

Individuals are :boldblue:`forward-looking` and so do not simply choose the
alternative with the highest immediate per-period utility. Instead, they take
future consequences of their actions into account and implement a
:boldblue:`policy` :math:`\pi \equiv (a_1^{\pi}(s_1), \dots, a_T^{\pi}(s_T))`.
A policy is a collection of :boldblue:`decision rules` :math:`a_t^{\pi}(s_t)`
that prescribe an action for any possible state :math:`s_t`. The implementation
of a policy generates a sequence of per-period utilities that depends on the
:boldblue:`objective transition probability distribution` :math:`p_t(s_t, a_t)`
for the evolution of state :math:`s_{t}` to :math:`s_{t+1}` induced by the model.
Individuals entertain :boldblue:`rational expectations` (Muth, 1961,
:cite:`Muth.1961`) so their subjective beliefs about the future agree with the
objective transition probabilities of the model.

:ref:`timing_events` depicts the timing of events in the model for two generic
periods. At the beginning of period t, an individual fully learns about the
immediate per-period utility of each alternative, chooses one of them, and
receives its immediate utility. Then the state evolves from :math:`s_t` to
:math:`s_{t+1}` and the process is repeated in :math:`t+1`.

.. _timing_events:

.. figure:: ../_static/images/timing_events.svg
  :width: 650
  :alt: Illustration timing of events

  Timing of events

Individuals face :boldblue:`uncertainty` (Gilboa, 2009, :cite:`Gilboa.2009`)
and discount future per-period utilities by an exponential discount factor
:math:`0 < \delta < 1` that parameterizes their time preference. Per-period
utilities are time-separable (Samuelson, 1937, :cite:`Samuelson.1937`).
Given an initial state :math:`s_1` individuals implement the policy :math:`\pi`
from the set of all feasible policies :math:`\Pi` that :boldblue:`maximizes the
expected total discounted utilities` over all :math:`T` periods.

 .. math::

    \underset{\pi \in \Pi}{\max} \, \mathbb{E}_{p^{\pi}} \left[ \sum_{t = 1}^T
    \delta^{t - 1} u(s_t, a_t^{\pi}(s_t)) \right],

where :math:`\mathbb{E}_{p^{\pi}}[\cdot]` denotes the expectation operator under
the probability measure :math:`p^{\pi}`. The decision problem is dynamic in the
sense that expected inter-temporal per-period utilities at a certain period
:math:`t` are influenced by past choices.

.. raw:: html

   <div
    <p class="d-flex flex-row gs-torefguide">
        <span class="badge badge-info">To Explanation</span></p>
    <p>The operationalization of the model allows to proceed with the calibration as
       described in<a href="mathematical_framework.html">
       Mathematical Framework</a> </p>
   </div>


.. rubric:: Footnotes

.. [#] For notational convenience we will omit the subscript :math:`a` whenever
       possible.
