The Economic Model
===================

Introduction
-------------

Keane and Wolpin (1997) develop a model in which an agent decides among :math:`M`
possible alternatives in each of :math:`T` (finite) discrete periods of time.
Alternatives are defined to be mutually exclusive and :math:`d_m(a) = 1` indicates that
alternative :math:`m` is chosen at time (age) :math:`a` and :math:`d_m(a)  = 0`
indicates otherwise. Associated with each choice is an immediate reward
:math:`R_m(S(a))` that is known to the agent at time :math:`a` but partly unknown from
the perspective of periods prior to :math:`a`. All the information known to the agent at
time :math:`a` that affects immediate and future rewards is contained in the state space
:math:`S(a)`.

We depict the timing of events below. At the beginning of period :math:`a` the agent
fully learns about all immediate rewards, chooses one of the alternatives and receives
the corresponding benefits. The state space is then updated according to the agent's
state experience and the process is repeated in :math:`a + 1`.

.. image:: ../images/timing.png

Agents are forward looking. Thus, they do not simply choose the alternative with the
highest immediate rewards each period. Instead, their objective at any time :math:`\tau`
is to maximize the expected rewards over the remaining time horizon:

.. math::
    \max_{\{d_m(a)\}_{m \in M}} E\left[ \sum_{\tau = a}^T \delta^{\tau - a} \sum_{m\in
    M}R_m(\tau)d_m(\tau)\Bigg| S(a)\right]

The discount factor :math:`0 > \delta > 1` captures the agent's preference for immediate
over future rewards. Agents maximize the equation above by choosing the optimal sequence
of alternatives :math:`\{d_m(a)\}_{m \in M}` for :math:`a = \tau, .., T`.

Within this more general framework, Keane and Wolpin (1997) consider the case where
agents are risk neutral and each period choose one of five sectors:

1. work in occupation A (in the paper: white-collar)
2. work in occupation B (in the paper: blue-collar)
3. work in occupation C (in the paper: the military)
4. continue education
5. remain at home


Details of the Notation
------------------------

In addition, individuals belong to one of K types that determine their base skill /
preference endowments in each of the sectors.

**Meaning of Subscripts**

================= ================================== ==========================
Notation          Interpretation                     Additional Information
================= ================================== ==========================
m                 sectors                            1: white-collar, 2: blue-collar,
                                                     3: military, 4: school, 5: home
k                 endowment type                     k = 1, 2, 3, 4 in the extended KW
                                                     model
a                 age /period from :math:`a_0` to A  from 16 to 65 in the extended KW
                                                     model
================= ================================== ==========================

Each of the five sectors is associated with a reward. The rewards depend on the
individual's base endowments and its experience in the different sectors, its age and
schooling level which together determine the individual's skill in each of the five
sectors.

**State-Space Variables**

================= ======================================== =============================
Notation          Interpretation                           Additional Information
================= ======================================== =============================
:math:`S_{a_0}`   state space at model start               :math:`S(16)` in the paper,
                                                           contains the type of the
                                                           individual
a                 age                                      from 16 to 65 in the extended
                                                           KW model
g                 schooling                                years of schooling
x                 work experience                          vector: one value for
                                                           white-collar, blue-collar and
                                                           military
:math:`d_1(a-1)`  whether occ. A was chosen last period
:math:`d_2(a-1)`  whether occ. B was chosen last period
:math:`d_4(a-1)`  whether education was chosen last period
================= ======================================== =============================


**Other Notation**

================= ===================================== ==========================
Notation          Interpretation                        Additional Information
================= ===================================== ==========================
:math:`d_m(a)`    dummy if sector m was chosen at age a
:math:`1[]`       indicator function                    1 if statement in [] is true,
                                                        0 else
================= ===================================== ==========================


The Reward Functions and Their Parameters
------------------------------------------

In what follows, we take a detailed look at the rewards of each alternative individuals
can choose. Some of the rewards of finishing high school and college are universal to
all sectors (:math:`\beta_1 1[g(a) \geq 12] + \beta_2 1[g(a) \geq 16]`).

In contrast to Keane & Wolpin (1997) here all linear indices only carry positive signs
as this is how parameters are specified in the model specification.

Occupation A
^^^^^^^^^^^^

The instantaneous reward associated with working in occupation A is given by the
following equation:

.. math::
    R_{1,k}(a) = r_1 \cdot e_{1,k}(a)
        &+ c_{1,1} \cdot 1[d_1(a-1) = 0] + c_{1,2} \cdot 1[x_1(a) = 0]\\
        &+ \alpha_1 + \beta_1 \cdot 1[g(a) \geq 12]\\
        &+ \beta_2 \cdot 1[g(a) \geq 16] + \beta_3 \cdot 1[x_3(a) = 1],

where  :math:`e_{1,k}(a)` are the current skills of an individual of type :math:`k` in
white-collar work. They evolve according to the following law:

.. math::
    e_{1,k}(a) = \exp\{\\
        e_{1,k}(a_0) &+ e_{1,11} \cdot g(a) + e_{1,12} \cdot 1[g(a) \geq 12]\\
                     &+ e_{1,13} \cdot 1[g(a) \geq 16] + e_{1,2} \cdot x_1(a)\\
                     &+ e_{1, 3} \cdot x^2_1(a) + e_{1,4} \cdot 1[x_1 > 0)]\\
                     &+ e_{1, 5} \cdot a + e_{1, 6} \cdot 1[a < 18]\\
                     &+ e_{1, 7} \cdot d_1(a-1) + e_{1,8} \cdot x_{2}(a)\\
                     &+ e_{1, 9} \cdot x_3(a)\\
    \} \cdot \exp[\epsilon_1(a)],

where :math:`\epsilon_1(a)` is the idiosyncratic, serially uncorrelated shock to the
individual's skills in age a. Note that in Appendix C of Keane & Wolpin (1997) the age
function is not necessarily linear (:math:`e_{1, 5}(a)`). However, from table 7 it
becomes clear that the age function was estimated to be linear so this is what is also
implemented in respy.

Furthermore, Keane & Wolpin (1997) only allow for a second degree polynomial function
for the same sector experience. Respy allows for both civilian sectors (occupation A and
B) to have quadratic effects on wages in both sectors.

.. include:: /economics/occ-a-params.rst

Occupation B
^^^^^^^^^^^^

The instantaneous reward associated with working in occupation B is analogous to that of
occupation A and given by the following function:

.. math::
    R_{2,k}(a) = r_2 \cdot e_{2,k}(a)
        &+ c_{2,1} \cdot 1[d_1(a-1) = 0] + c_{2,2} \cdot 1[x_2(a) = 0]\\
        &+ \alpha_2 + \beta_1 \cdot 1[g(a) \geq 12]\\
        &+ \beta_2 \cdot 1[g(a) \geq 16] + \beta_3 \cdot 1[x_3(a) = 1],


where :math:`e_{2,k}(a)` are the current skills of the individual in blue-collar work.
They evolve according to the following law:

.. math::
    e_{2,k}(a) = \exp\{\\
        e_{2,k}(a_0) &+ e_{2,11} \cdot g(a) + e_{2,12} \cdot 1[g(a) \geq 12]\\
                     &+ e_{2,13} \cdot  1[g(a) \geq 16] + e_{2,2} \cdot x_2(a)\\
                     &+ e_{2, 3} \cdot  x^2_2(a) + e_{2,4} \cdot  1[x_2 > 0)]\\
                     &+ e_{2, 5} \cdot a + e_{2, 6} \cdot  1[a < 18]\\
                     &+ e_{2, 7} \cdot d_1(a-1) + e_{2,8} \cdot x_{1}(a)\\
                     &+ e_{2, 9} \cdot  x_3(a)\\
    \} \cdot \exp[\epsilon_2(a)]

The same comments as for occupation A apply.

.. include:: /economics/occ-b-params.rst

Occupation C
^^^^^^^^^^^^

The instantaneous reward associated with working in occupation C is different from
occupation A and B in several aspects:

- In the skill technology function:
    - all types have the same base skill endowment in the military (can be seen in table
      7, although the state space on page 521 suggests something different).
    - no sheep skin effects for finishing high school or college (:math:`e_{m,12},
      e_{m,13}`)
    - no effects of civilian sector experiences on the military skills
    - no effect of switching to the military from another sector (:math:`e_{m, 7}`)
- In the general reward function:
    - The non-pecuniary reward (:math:`\alpha`) is multiplicative and not additive (see
      note below on how it is nevertheless identified)
    - The non-pecuniary reward (:math:`\alpha`) varies with age (but skills are not
      allowed to change with age)
    - no penalty of leaving the military prematurely (:math:`\beta_3 \cdot 1[x_3(a) =
      1]`)
    - There is no reward or penalty for never having worked in the military before
      (:math:`1[d_3(a-1) = 0]`)

The reward function is given by the following equation:

.. math::
    R_{3,k}(a) = \exp[\alpha_3(a)] \cdot r_3 \cdot e_3(a) + c_{3,2} \cdot 1[x_3(a) = 0]
    + \\ & \beta_1 \cdot 1[g(a) \geq 12] + \beta_2 \cdot 1[g(a) \geq 16] \cdot
    exp\{\epsilon_3(a)\}


where :math:`e_{3}(a)` are the current skills of the individual in military work. They
evolve according to the following law:

.. math::
    e_{3}(a) = exp \{ & e_3(a_0) + e_{3,1} \cdot g(a) + e_{3,2} \cdot x_3(a) + e_{3,3}
    \cdot x_3^2(a) + \\ & e_{3,4} \cdot 1[x_3 > 0] + e_{3,5}(a) + e_{3, 6} \cdot 1[a <
    18]\}

.. include:: /economics/occ-c-params.rst

Note that the coefficients of the skill technology function are identified by the wage
data separately from the other coefficients. Thus, both a constant factor (:math:`r_3
\cdot exp(e_3(a_0))`) in the wage part and a constant factor (:math:`\alpha_{3,c}`) in
the non-pecuniary part can be identified. However, :math:`r_3 \cdot exp(e_3(a_0))` can
only be estimated jointly.

.. note::

    In the text of the paper it appears that there is a cost of leaving the military
    sector prematurely that is effective in all sectors but the military
    (:math:`\beta_3`). The result table  (Table 7), however, suggests that a similar
    channel was captured by a benefit of not leaving the military sector prematurely
    that is only effective in the military sector. We implement this second version.

Furthermore, from table 7 it becomes again apparent that the unspecified functions
:math:`\alpha (a)` and :math:`e_{3,5}(a)` are linear in age and
:math:`exp\{\epsilon_3(a)\}` is missing in the formulas of Appendix C but mentioned in
the text.

Schooling
^^^^^^^^^

The instantaneous utility of schooling is:

.. math::
    R_{4,k}(a) = & e_{4,k}(a_0) + tc_1 \cdot 1[g(a) \geq 12]
    + tc_2 \cdot 1[g(a) \geq 16] + \\
    & rc_1 \cdot 1[d_4(a-1) = 0, g(a) < 12] + \\
    & rc_2 \cdot 1[d_4(a-1) = 0, g(a) \geq 12] + \\
    & \beta_1 \cdot 1[g(a) \geq 12] + \beta_2 \cdot 1[g(a) \geq 16] + \\
    & \beta_3 \cdot 1[x_3(a) = 1] + \gamma_{4,1} \cdot a + \\
    & \gamma_{4,2} \cdot 1[a < 18]+ \epsilon_4(a)

**Education Sector Parameters**

.. include:: /economics/occ-edu-params.rst

There are no skills in education that evolve over time. However, each type has a
different base endowment :math:`e_{4, k}(a_0)` that affects how much utility this type
derives from going to school.

Home Sector
************

The instantaneous utility of staying home is:

.. math::
    R_{5,k}(a) = & e_{5,k}(a_0) + \beta_1 \cdot 1[g(a) \geq 12]
    + \beta_2 \cdot 1[g(a) \geq 16] + \\
    & \beta_3 \cdot 1[x_3(a) = 1] + \gamma_{5, 1} \cdot 1[18 \leq a \leq 20]
    + \gamma_{5, 2} 1[a \geq 21] + \epsilon_5(a)

**Home Sector Parameters**

.. include:: /economics/occ-home-params.rst

There are no skills in home production that evolve over time. However, each type has a
different base endowment :math:`e_{5, k}(a_0)` that affects how much utility this type
derives from staying home.


The Role of Shocks in the Model
---------------------------------

The :math:`\epsilon_{m}(a)` are alternative-specific shocks to occupational
productivity, to the consumption value of schooling, and to the value of non-market
time. The productivity and taste shocks follow a five-dimensional multivariate normal
distribution with mean zero and covariance matrix :math:`\Sigma = [\sigma_{m, m'}]`. The
realizations are independent across time.

Given the structure of the reward functions and the agent's objective, the state space
at time :math:`a` is:

.. math::

    S(a) = \{& S(a_0), a, g(a), x_{1}(a), x_{2}(a), x_{3}(a), \\
    & d_1(a - 1), d_2(a - 1), d_4(a - 1),\\
    & \epsilon_{1}(a), \epsilon_{2}(a),\epsilon_{3}(a),\epsilon_{4}(a),
    \epsilon_{5}(a)\}.

It is convenient to denote its observable elements as :math:`\bar{S}(a)`. The elements
of :math:`S(a)` evolve according to:

.. math::

    x_{1}(a+1)  &= x_{1}(a) + d_1(a)

    x_{2}(a+1) &= x_{2}(a) + d_2(a)

    x_{3}(a+1) &= x_{3}(a) + d_3(a)

    g(a+1)    &= g(a) + d_4(a)

    f(\epsilon_{t+1}\mid S(t), d_k(t)) &= f(\epsilon_{t+1}\mid \bar{S}(t), d_k(t)),
    \forall k \in \{1, 2, 3, 4, 5\}

where the last equation reflects the fact that the :math:`\epsilon_{k, t}`'s are
serially independent. We set :math:`x_{1}(a_0) = x_{2}(a_0) = x_{3}(a_0) = 0` as the
initial conditions.
