The Economic Model
===================

.. todo::
    Change equations and functions to have type shifts instead of group specific constants

Introduction
-------------

Keane and Wolpin (1997) develop a model in which an agent decides among :math:`M` possible alternatives in each of :math:`T` (finite) discrete periods of time.  Alternatives are defined to be mutually exclusive and :math:`d_m(a) = 1` indicates that alternative :math:`m` is chosen at time (age) :math:`a` and :math:`d_m(a)  = 0` indicates otherwise. Associated with each choice is an immediate reward :math:`R_m(S(a))` that is known to the agent at time :math:`a` but partly unknown from the perspective of periods prior to :math:`a`. All the information known to the agent at time :math:`a` that affects immediate and future rewards is contained in the state space :math:`S(a)`.

We depict the timing of events below. At the beginning of period :math:`a` the agent fully learns about all immediate rewards, chooses one of the alternatives and receives the corresponding benefits. The state space is then updated according to the agent's state experience and the process is repeated in :math:`a + 1`.

.. image:: images/timing.png

Agents are forward looking. Thus, they do not simply choose the alternative with the highest immediate rewards each period. Instead, their objective at any time :math:`\tau` is to maximize the expected rewards over the remaining time horizon:

.. math::
    \max_{\{d_m(a)\}_{m \in M}} E\left[ \sum_{\tau = a}^T \delta^{\tau - a} \sum_{m\in M}R_m(\tau)d_m(\tau)\Bigg| S(a)\right]

The discount factor :math:`0 > \delta > 1` captures the agent's preference for immediate over future rewards. Agents maximize the equation above by choosing the optimal sequence of alternatives
:math:`\{d_m(a)\}_{m \in M}` for :math:`a = \tau, .., T`.

Within this more general framework, Keane and Wolpin (1997) consider the case where agents are risk neutral and each period choose one of five sectors:

1. work in occupation A (in the paper: white-collar)
2. work in occupation B (in the paper: blue-collar)
3. work in occupation C (in the paper: the military)
4. continue education
5. remain at home


Details of the Notation
------------------------

In addition, individuals belong to one of K types that determine their base endowments.

**Meaning of Subscripts**

=================       =====================                   ==========================
Notation                Interpretation                             Additional Information
=================       =====================                   ==========================
m                       sectors                                 1: white-collar, 2: blue-collar, 3: military, 4: school, 5: home
k                       endowment type                          k = 1, 2, 3, 4 in the extended KW model
a                       age /period                             from 16 to 65 in the extended KW model
=================       =====================                   ==========================

Each of the five sectors is associated with a reward. The rewards depend on the individual's base endowments and its experience in the different sectors, its age and schooling level which together determine the individual's skill in each of the five sectors.

**State-Space Variables**

=================       ===========================================    ==========================================
Notation                Interpretation                                 Additional Information
=================       ===========================================    ==========================================
:math:`S_0`             state space at model start                     :math:`S(16)` in the paper, contains the type of the individual
a                       age                                            from 16 to 65 in the extended KW model
g                       schooling                                      years of schooling
x                       work experience                                vector: one value for white-collar, blue-collar and military
:math:`d_1(a-1)`        whether occ. A was chosen last period
:math:`d_2(a-1)`        whether occ. B was chosen last period
:math:`d_4(a-1)`        whether education was chosen last period
=================       ===========================================    ==========================================


**Other Notation**

=================       =====================================       ==========================
Notation                Interpretation                                 Additional Information
=================       =====================================       ==========================
:math:`d_m(a)`          dummy if sector m was chosen at age a
:math:`1[]`             indicator function                          1 if statement in [] is true, 0 else
=================       =====================================       ==========================


The Reward Functions and Their Parameters
------------------------------------------

We now take a look at the reward functions of the five sectors:

Occupation A
*************

The instantaneous reward associated with working in occupation A is given by the following equation:

.. math::
    R_{1,k}(a) = r_1 \cdot e_{1k}(a) - c_{11} \cdot 1[d_1(a-1) = 0] - c_{12} \cdot 1[x_1(a) = 0] + \alpha_1 + \beta_1 1[g(a) \geq 12] + \beta_2 1[g(a) \geq 16] + \beta_3 1[x_3(a) = 1],

where  :math:`e_{1k}(a)` are the current skills of the individual in white-collar work. They evolve according to the following law:

.. math::
    e_{1k}(a) = exp\{e_{m,6} + e_{m11}g(a) + e_{m,12}1[g(a) \geq 12] + e_{m,13} 1[g(a) \geq 16] + e_{m,2}x_m(a) - e_{m, 3} x^2_m(a) + e_{m,4} 1[x/m > 0)] + e_{m, 5}(a) + e_{m, 6} 1[a < 18] + e_{m, 7} d_m(a-1) + e_{m,8}x_{m'}(a) + e_{m, 9} x_3(a) \}

.. include:: occ_a_params.rst

Occupation B
*************

The instantaneous reward associated with working in occupation B is given by the following function:

.. math::
    R_{2,k}(a) = r_2 \cdot e_{2k}(a) - c_{21} \cdot 1[d_1(a-1) = 0] - c_{22} \cdot 1[x_2(a) = 0] + \alpha_2 + \beta_1 1[g(a) \geq 12] + \beta_2 1[g(a) \geq 16] + \beta_3 1[x_3(a) = 1],


where :math:`e_{2k}(a)` are the current skills of the individual in blue-collar work. They evolve according to the following law:

.. math::
    e_{2k}(a) = ...

.. include:: occ_b_params.rst

Occupation C
*************

The instantaneous reward associated with working in occupation C is given by the following function:

.. math::
    R_{3,k}(a) = ...,

where :math:`e_{3k}(a)` are the current skills of the individual in military work. They evolve according to the following law:

.. math::
    e_{3k}(a) = ...

.. include:: occ_c_params.rst

Note that the coefficients of the skill technology function are identified by the wage data separately from the other coefficients. Thus, both a constant factor (:math:`r_3 \cdot exp(e_3(16))`) in the wage part and a constant factor (:math:`\alpha_{3,c`) in the non-pecuniary part can be identified. However, :math:`r_3 \cdot exp(e_3(16))` can only be estimated jointly.

.. Note::
    In the text of the paper it appears that there is a cost of leaving the military sector prematurely that is effective in all sectors but the military. The result table  (Table 7), however, suggests that a similar channel was captured by a benefit of not leaving the military sector prematurely that is only effective in the military sector. We implement this second version.

Schooling
***********

The instantaneous utility of schooling is:

.. math::
    R_{4,k}(a) = ...

**Education Sector Parameters**

.. include:: occ_edu_params.rst

There are no skills in education that evolve over time. However, each type has a different base endowment :math:`e_{4k}(16)` that affects how much utility this type derives from going to school.

Home Sector
************

The instantaneous utility of staying home is:

.. math::
    R_{3,k}(a) = ...

**Home Sector Parameters**

.. include:: occ_home_params.rst

There are no skills in home production that evolve over time. However, each type has a different base endowment :math:`e_{5k}(16)` that affects how much utility this type derives from staying home.


The Role of Shocks in the Model
---------------------------------

.. todo::
    Update to KW 97!

The :math:`\epsilon_{ma}`'s are alternative-specific shocks to occupational productivity, to the consumption value of schooling, and to the value of non-market time. The productivity and taste shocks follow a four-dimensional multivariate normal distribution with mean zero and covariance matrix :math:`\Sigma = [\sigma_{ij}]`. The realizations are independent across time. We collect the parametrization of the reward functions in :math:`\theta = \{\alpha_1, \alpha_2, \beta, \gamma, \Sigma\}`.

Given the structure of the reward functions and the agents objective, the state space at time :math:`a` is:

.. math::

    S(a) = \{S(a_0), a, g(a), x_{1,a},x_{2,a}, x_{3,a}, d_1(a - 1), d_2(a - 1), d_4(a - 1),\epsilon_{1}(a),\epsilon_{2}(a),\epsilon_{3}(a),\epsilon_{4}(a), \epsilon_{5}(a)\}.

It is convenient to denote its observable elements as :math:`\bar{S}(a)`. The elements of :math:`S(a)` evolve according to:

.. math::

    x_{1,a+1}  &= x_{1a} + d_1(a)

    x_{2,a+1} &= x_{2a} + d_2(a)

    x_{3,a+1} &= x_{3a} + d_3(a)

    g(a+1)    &= g(a) + d_4(a)

    f(\epsilon_{t+1}\mid S(t), d_k(t)) &= f(\epsilon_{t+1}\mid \bar{S}(t), d_k(t)),

where the last equation reflects the fact that the :math:`\epsilon_{kt}`'s are serially independent. We set :math:`x_{1t} = x_{2t} = 0` as the initial conditions.
