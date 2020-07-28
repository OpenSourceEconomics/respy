.. _economic_model:

The Economic Framework
======================


.. role:: boldblue

To illustrate the usage of ``respy`` to solve finite-horizon discrete choice
dynamic programming (DCDP) we will outline the human capital model
conceptualized in the work of Keane and Wolpin (1997, :cite:`Keane.1997`).
Under uncertainty individuals face sequential decisions
whether to engage in a certain occupation, attend school, or to enjoy leisure.
Keane and Wolpin (1997, :cite:`Keane.1997`) build upon a basic human capital
model and extend it to optimally fit empirical data on schooling and
occupational choice.

Individuals enter the model at age :math:`16` and take decisions until
they reach the :boldblue:`final period` :math:`T`. At each point in time
:math:`t \in \{ 16, \dots, T \}` individuals observe the
:boldblue:`state of the world` and choose :math:`a_{it}` one of five
mutually exclusive alternatives :math:`a \in \mathcal{A} = \{1, \dots, 5\}`.
The choice has two consequences:

    - The state of the world evolves :math:`S_{it} \to S_{i,t+1}`.
    - Individuals receive an immediate :boldblue:`per-period reward`
      :math:`R_a(S_{it})` associated with the choice :math:`a_{it}`.

In general, the per-period rewards summarize all cost and benefits that come
along with the choice :math:`a_{it}` of a specific alternative
:math:`a \in \mathcal{A}`. The reward itself depends on a subset of the state
of the world :math:`S_{it}` observed by the individual. Choices modeled in
Keane and Wolpin 1997, :cite:`Keane.1997`) and associated rewards are
summarized in Table :ref:`choice_explanation_reward` and explained in Section
:ref:`explaining_rewards`. Choices :math:`a \in \mathcal{A}^w = \{1, 2, 3 \}`
are labeled working alternatives, and choices
:math:`a \in  \mathcal{A}^{nw} = \{4,5 \}` are labeled non-working
alternatives, and :math:`\mathcal{A} = \mathcal{A}^{w} \cup \mathcal{A}^{nw}`.

.. _choice_explanation_reward:

.. csv-table:: Overview of choices and associated rewards in the
               Keane and Wolpin (1997, :cite:`Keane.1997`) model
   :header: "Choice", "Explanation", "Reward"


   ":math:`a_{it}` = 1", "Work in white-collar occupation
   ", ":math:`R_1(S_{it}) = W_{it}(1)`"
   ":math:`a_{it}` = 2", "Work in blue-collar occupation
   ", ":math:`R_2(S_{it}) = W_{it}(2)`"
   ":math:`a_{it}` = 3", "Service in the military
   ", ":math:`R_3(S_{it}) = W_{it}(3)`"
   ":math:`a_{it}` = 4", "Attend school or university
   ", ":math:`R_4(S_{it}) = e_i(4) + \beta_{tc_1} 1(h_{it} \geq 12) \\
   + \beta_{tc_2} 1(h_{it} \geq 16) + \epsilon_{it}(4)`"
   ":math:`a_{it}` = 5", "Stay at home
   ", ":math:`R_5(S_{it}) = e_i(5) +\epsilon_{it}(5)`"


Choosing and remaining in a certain occupation leads to the accumulation of
occupation-specific :boldblue:`experience` :math:`k_{it}(a)`.
Remaining in school adds to the years of completed schooling :math:`h_{it}`.

.. rst-class:: centerblue

   In ``respy`` choices that allow for accumulation of experience
   are of the form ``exp_{choice}``. Whenever the individual remains in a
   certain occupation and the state :math:`S_{it}` evolves to
   :math:`S_{i, t+1}` experience is increased by one.

When entering the model individuals are assumed to be heterogeneous in their
:boldblue:`endowments` which are composed of of prior human capital
accumulation, innate talents, and abilities. Some individuals may be
very talented at school while others are gifted with aptitude to applied
work, i.e. some individuals have a comparative advantage
among the different choice alternatives.

This type of :boldblue:`heterogeneity` introduced through endowment types
:math:`k = 1, \dots, K`. The vector of endowments for type :math:`k` is given
by :math:`e_{ik}  = (e_{ik}(1), e_{ik}(2), e_{ik}(3), e_{ik}(4), e_{ik}(5))`
for :math:`k = 1, \dots, K`. Individuals know their endowment type.
The researcher knows only that there exist :math:`K` types
but the endowment heterogeneity is unobserved.

The :boldblue:`occupation-specific skill level` of an individual
:math:`x_{it}(a)` is a function of the individual experience :math:`k_{it}(a)`
, completed years of schooling :math:`h_{it}`, and the individual endowment
:math:`e_{ik}(a)`. While endowments are pre-determined [#]_ the individual
can invest into its human capital and increase its skill level
:math:`x_{it} = \{ x_{it}(a) \text{ for } a \in \mathcal{A} \}`.
The particular form is described given in
Equation :eq:`OccupationSpecificSkillLevel`.

We assume individuals to be :boldblue:`forward-looking` and to entertain
:boldblue:`rational expectations` (Lucas, 1972, :cite:`Lucas.1972`)
while following the objective to maximize their expected lifetime
utility. [#]_ Individuals do not simply choose the alternative with the
highest immediate reward but take the future consequences of their current
actions into account. Individuals are assumed to be :boldblue:`risk-neutral`,
hence have linear instantaneous utility functions.

The decision problem is :boldblue:`dynamic` in the sense that the decision
to work in a certain occupation leads to accumulation of experience and in
turn to higher occupation-specific rewards. We will call the sequence of
optimal choices [#]_ the :boldblue:`policy` :math:`\pi_i`.
How individuals derive this policy is explained in :ref:`solution_model`.

.. _explaining_rewards:

Explaining Rewards
------------------

Rewards for Working Alternatives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The :boldblue:`reward for the working alternatives` is given by the
occupation-specific wage :math:`W_{it}(a)`, :math:`a \in \mathcal{A}^w`.
The wage is modeled as product of the occupation-specific skill level
:math:`x_{it}(a)` and the occupation-specific market (equilibrium)
rental price :math:`r_a`: [#]_

.. math::

   W_{it}(a) = r_a \cdot x_{it}(a).

As common in a standard human capital formulation, the
:boldblue:`occupation-specific skill level` is a function of the
successfully completed years of schooling, :math:`h_{it}`,
and the accumulated work experience from a certain occupation,
:math:`k_{it}(a)` disturbed by a technology shock :math:`\epsilon_{it}(a)`.
The accumulated work experience at :math:`t=16` assumed to be zero,
i.e. :math:`k_{i, 16}(a) = 0` for :math:`a \in \mathcal{A}^w`.
For ease of notation we will not distinguish between different
endowment types :math:`k`. The occupation-specific skill level is
composed as: [#]_

.. math::
   :label: OccupationSpecificSkillLevel

    x_{it}(a) = \exp( e_i(a) + \beta_{a1} \cdot h_{it} + \beta_{a2} \cdot
    k_{it}(a) + \beta_{a3} \cdot (k_{it}(a))^2 + \epsilon_{it}(a)).

The exponential form carries a characteristic feature: Investment into
work experience in a particular occupation or investment into schooling are
particularly productive if the respective initial endowments are high. [#]_

:boldblue:`Occupation-specific technology shocks` are denoted by
:math:`\epsilon_{it}^w = \{ \epsilon_{it}(a): a \in \mathcal{A}^w \}`.
The shocks are assumed to be serially uncorrelated and to follow a
multivariate normal distribution with zero mean and unrestricted
variance matrix, :math:`\epsilon_{it}^w \sim \mathcal{N}_3(0, \Sigma^w)`.

Taken together the occupation-specific wage :math:`W_{it}(a)` can be written as

.. math::
   :label: OccupationSpecificWage

   W_{it}(a) =   r_a \cdot \exp( e_i(a) + \beta_{a1} \cdot h_{it} +
   \beta_{a2} \cdot k_{it}(a) + \beta_{a3} \cdot (k_{it}(a))^2
   + \epsilon_{it}(a)).

This formulation is a standard log-wage equation with constant term
:math:`\ln(r_a) + e_i(a)`. It is noteworthy that the payoff variables
:math:`W_{it}(a)` is not independent of the random technology shocks.
More specifically, :math:`\epsilon_{it}(1), \epsilon_{it}(2)`, and
:math:`\epsilon_{it}(3)` have an effect on the rewards and
hence on the choices of the individuals.

Rewards for Non-Working Alternatives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The :boldblue:`reward for schooling` diverts from the ''pure'' human capital
investment model by introducing an additional effort cost. It is given by

.. math::
   :label: RewardSchooling

   R_4(S_{it}) = e_i(4) + \beta_{tc_1} \cdot \mathbb{1}(h_{it} \geq 12)
   + \beta_{tc_2} \cdot \mathbb{1}(h_{it} \geq 16) + \epsilon_{it}(4).

Parameters :math:`\beta_{tc1}` and :math:`\beta_{tc2}` denote
:boldblue:`direct tuition cost` associated with attending college and
graduate school, respectively. [#]_ In some cross-section analysis
the direct effort cost may be not discernible at all, e.g.
in government-provided schooling systems the tuition costs are
merely the same for every student.

The :boldblue:`effort cost` is composed of the initial endowment
:math:`e_i(4)` and a taste shock
:math:`\epsilon_{it}(4) \sim \mathcal{N}(0, \sigma_4^2)`.
The taste shock realizes at the beginning of each period
:math:`t` and leads to a fluctuation in the reward of schooling. [#]_

The :boldblue:`reward from being at home` is given by a ''home-related''
endowment and normally distributed taste shocks
:math:`\epsilon_{it}(5) \sim \mathcal{N}(0, \sigma_5^2)`
that fluctuate with age.

.. math::
   :label: RewardHome

    R_5(s_{it}) = e_i(5) +\epsilon_{it}(5).

At this point it may be noteworthy that - in the model - the value of
non-working rewards is interpreted as nominal value, i.e. in dollars.


.. _economic_interpretation:

Economic Interpretation
-----------------------

Individuals can investment into their human capital in possible ways.
Deciding for a working alternative increases their experience in a
certain occupation and hence their future earning prospects in those.
Attending school on the other hand builds human capital to land better jobs.
However, schooling has a ''bitter side taste'' since it entails foregone
earnings and work experience, or foregone leisure.

The optimal path of human capital investment is an individual-specific
combination of attending school and going to work.
The reward structure is key to understand how individuals choose their
career paths. The type-endowments pre-determine the career paths
to a certain extent. Individuals that are endowed with school-related
abilities will be inclined to stay relatively longer in school.
Indeed Keane and Wolpin (1997, p. 497, :cite:`Keane.1997`) find
in their extended human capital model that a large fraction of
inequality in career paths, lifetime earnings, and lifetime utility can be
attributed to the different skill endowment at age :math:`16`.

However, individuals invest into their human capital under uncertainty:
periodically occurring technology and taste shocks alter the rewards
from any alternative. Their investment decisions require to take into
account those shocks. The model stipulates that individuals know the
exact probability distribution of those shocks and act in an expected
utility framework.

.. rubric:: Footnotes

.. [#] The human capital investment process has already started before
       individuals enter the model. For example, parents may have already
       invested into their children in order to foster their development.

.. [#] In the model outlined by Keane and Wolpin (1997, :cite:`Keane.1997`)
       individuals operate in an expected utility framework.

.. [#] Optimal in the sense that it maximizes the lifetime utility,
       or in general the objective function, of the individual.

.. [#] Note that the model is a partial-equilibrium model
       and hence takes the rental price as given.

.. [#] The specification follows closely Mincer (1958, :cite:`Mincer.1958`).

.. [#] Consequently, different endowment types can be
       introduced to create a persistent choice pattern.

.. [#] The implementation in ``respy`` allows only for additive terms.
       Hence we redefined the original expression from Keane and
       Wolpin (1997, :cite:`Keane.1997`)
       :math:`R_4(s_{it}) = e_i(4) - \beta'_{tc_1} 1(h_{it} \geq 12)
       - \beta'_{tc_2} 1(h_{it} \geq 16) + \epsilon_{it}(4)` by replacing
       :math:`\beta'_{tc}= - \beta_{tc}`.

.. [#] Through the different endowment types a similar mechanism as in the
       working alternatives is at place. Types with a higher endowment
       of school-related abilities will be more inclined to
       extent their schooling.
