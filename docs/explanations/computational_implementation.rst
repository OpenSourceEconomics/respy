.. _computational_implementation:

Computational Implementation
============================

.. role:: boldblue

We will present the :boldblue:`computational implementation` of the model.

.. raw:: html

   <div
    <p class="d-flex flex-row gs-torefguide">
        <span class="badge badge-info">To Explanation</span></p>
    <p>A list of the parameter-covariates combination as implemented in <b><i>respy</i></b>
       is presented in<a href="parameterization.html">
       Parameterization</a> </p>
   </div>

We follow individuals over their working life from young adulthood at age 16
to retirement at age 65. The decision period :math:`t = 16, \dots, 65`  is
a school year and individuals decide :math:`a\in\mathcal{A}` whether to
work in a blue-collar or white-collar occupation (:math:`a = 1, 2`),
to serve in the military :math:`(a = 3)`, to attend school :math:`(a = 4)`,
or to stay at home :math:`(a = 5)`.

.. figure:: ../_static/images/event_tree.pdf
  :width: 600


Individuals are initially heterogeneous. They differ with respect to their
initial level of completed schooling :math:`h_{16}` and have one of four
different :math:`\mathcal{J} = \{1, \dots, 4\}` alternative-specific skill
endowments
:math:`\boldsymbol{e} = \left(e_{j,a}\right)_{\mathcal{J} \times \mathcal{A}}`.

--------------------------------------------------------------------------------

.. raw:: html

   <div
    <p class="d-flex flex-row gs-torefguide">
        <span class="badge badge-info">To How-to guide</span></p>
    <p>An explanation how to set and change initial conditions in <b><i>
    respy</i></b> is provided in<a href="../how_to_guides/initial_conditions.html">
    Initial Conditions</a> </p>
   </div>

--------------------------------------------------------------------------------

The :boldblue:`per-period utility` :math:`u_a(\cdot)` of each alternative
consists of a non-pecuniary utility :math:`\zeta_a(\cdot)` and, at least for the
working alternatives, an additional wage component :math:`w_a(\cdot)`.
Both depend on the level of human capital as measured by their
alternative-specific skill endowment :math:`\boldsymbol{e}`,
years of completed schooling :math:`h_t`, and occupation-specific work
experience :math:`\boldsymbol{k_t} = \left(k_{a,t}\right)_{a\in\{1, 2, 3\}}`.
The immediate utility functions are influenced by last-period choices
:math:`a_{t -1}` and alternative-specific productivity shocks
:math:`\boldsymbol{\epsilon_t} = \left(\epsilon_{a,t}\right)_{a\in\mathcal{A}}`
as well. Their general form is given by:

.. math::
    u_a(\cdot) =
    \begin{cases}
    \zeta_a(\boldsymbol{k_t}, h_t, t, a_{t -1})  + w_a(\boldsymbol{k_t}, h_t,
     t, a_{t -1}, e_{j, a}, \epsilon_{a,t})
     & \text{if}\, a \in \{1, 2, 3\}  \\
    \zeta_a(\boldsymbol{k_t}, h_t, t, a_{t-1}, e_{j,a}, \epsilon_{a,t})
    &  \text{if}\, a \in \{4, 5\}.
    \end{cases}

Work experience :math:`\boldsymbol{k_t}`  and years of completed schooling
:math:`h_t` evolve deterministically.

.. math::
   k_{a,t+1} =
   k_{a,t} + \mathbb{1}[a_t = a]  &\qquad \text{if}\, a \in \{1, 2, 3\} \\
   h_{t + 1\phantom{,a}} = h_{t\phantom{,a}} +   \mathbb{1}[a_t = 4]  &\qquad

The :boldblue:`productivity shocks` are uncorrelated across time and follow a
multivariate normal distribution with mean :math:`\boldsymbol{0}` and
covariance matrix :math:`\boldsymbol{\Sigma}`. Given the structure of the
utility functions and the distribution of the shocks, the state at time
:math:`t` is summarized by the object
:math:`s_t = \{\boldsymbol{k_t}, h_t, t, a_{t -1},
\boldsymbol{e},\boldsymbol{\epsilon_t}\}`.

--------------------------------------------------------------------------------

.. raw:: html

   <div
    <p class="d-flex flex-row gs-torefguide">
        <span class="badge badge-info">To Reference guide</span></p>
    <p>A sophisticated account of the complete state space is integrated in
       <b><i>respy</i></b>. The particular implementation is described in
       <a href="../autoapi/respy/state_space/index.html">respy.state_space</a></p>
   </div>


--------------------------------------------------------------------------------

Empirical and theoretical research from specialized disciplines within
economics informs the exact specification of :math:`u_a(\cdot)`.
We now discuss each of its components in detail.

Non-pecuniary utility
---------------------
We present the parameterization of the non-pecuniary utility for
all five alternatives.

Blue-collar
^^^^^^^^^^^
Equation :eq:`NonWageBlueCollar` shows the parameterization of the
non-pecuniary utility from working in a blue-collar occupation.

.. math::
   :label: NonWageBlueCollar


   \zeta_{1}(\boldsymbol{k_t}, h_t, a_{t-1})  = \alpha_1  &+ c_{1,1}
   \cdot \mathbb{1}[a_{t-1} \neq 1] + c_{1,2} \cdot \mathbb{1}[k_{1,t} = 0] \\
   & + \vartheta_1 \cdot \mathbb{1}[h_t \geq 12] + \vartheta_2 \cdot
   \mathbb{1}[h_t \geq 16] + \vartheta_3 \cdot \mathbb{1}[k_{3,t} = 1]

A constant :math:`\alpha_1` captures the net monetary-equivalent of on the
job amenities. The non-pecuniary utility includes mobility and search costs
:math:`c_{1,1}`, which are higher for individuals who never worked in a
blue-collar occupation before :math:`c_{1,2}`. The non-pecuniary utilities
capture returns from a high school :math:`\vartheta_1` and a college
:math:`\vartheta_2` degree. Additionally, there is a detrimental effect of
leaving the military early after one year :math:`\vartheta_3`.

White-collar
^^^^^^^^^^^^
The non-pecuniary utility from working in a white-collar occupation is
specified analogously. Equation :eq:`UtilityWhiteCollar` shows its
parameterization.

.. math::
   :label: UtilityWhiteCollar

   \zeta_{2}( \boldsymbol{k_t}, h_t, a_{t-1} ) = \,\alpha_2 & + c_{2,1}
   \cdot \mathbb{1}[a_{t-1} \neq 2] + c_{2,2} \cdot \mathbb{1}[k_{2,t} = 0]\\
   & + \vartheta_1 \cdot \mathbb{1}[h_t \geq 12] + \vartheta_2 \cdot
   \mathbb{1}[h_t \geq 16] + \vartheta_3 \cdot \mathbb{1}[k_{3,t} = 1]


Military
^^^^^^^^
Equation :eq:`UtilityMilitary` shows the parameterization of the
non-pecuniary utility from working in the military.

.. math::
   :label: UtilityMilitary

   \zeta_{3}( k_{3.t}, h_t)  = c_{3,2} \cdot \mathbb{1}[k_{3,t} = 0] +
   \vartheta_1 \cdot \mathbb{1}[h_t \geq 12] + \vartheta_2 \cdot
   \mathbb{1}[h_t \geq 16]


Search costs :math:`c_{3, 1} = 0` are absent but there is a mobility cost if
an individual has never served in the military before :math:`c_{3,2}`.
Individuals still experience a non-pecuniary utility from finishing
high-school :math:`\vartheta_1` and college :math:`\vartheta_2`.


School
^^^^^^
Equation :eq:`UtilitySchooling` shows the parameterization of the
non-pecuniary utility from schooling.

.. math::
   :label: UtilitySchooling

   \zeta_4(k_{3,t}, h_t, t, a_{t-1}, e_{j,4}, \epsilon_{4,t})  = e_{j,4} & +
   \beta_{tc_1} \cdot \mathbb{1}[h_t \geq 12] + \beta_{tc_2}
   \cdot \mathbb{1}[h_t \geq 16]   \\\nonumber
   & + \beta_{rc_1} \cdot \mathbb{1}[a_{t-1} \neq 4, h_t < 12] + \beta_{rc_2}
   \cdot \mathbb{1}[a_{t-1} \neq 4, h_t \geq 12] \\\nonumber
   & + \gamma_{4,4} \cdot t + \gamma_{4,5} \cdot \mathbb{1}[t < 18] \\\nonumber
   & + \vartheta_1 \cdot \mathbb{1}[h_t \geq 12] + \vartheta_2 \cdot
   \mathbb{1}[h_t \geq 16] + \vartheta_3 \cdot \mathbb{1}[k_{3,t} = 1]\\
   & + \epsilon_{4,t}

There is a direct cost of attending school such as tuition for continuing
education after high school :math:`\beta_{tc_1}` and college
:math:`\beta_{tc_2}`. The decision to leave school is reversible,
but entails adjustment costs that differ by schooling category
(:math:`\beta_{rc_1}, \beta_{rc_2}`). Schooling is defined as time spent
in school and not by formal credentials acquired. Once individuals reach
a certain amount of schooling, they acquire a degree.
There is no uncertainty about grade completion (Altonji, 1993,
:cite:`Altonji.1993`) and no part-time enrollment. Individuals value the
completion of high-school and graduate school
(:math:`\vartheta_1, \vartheta_2`).

Home
^^^^
Equation :eq:`UtilityHome` shows the parameterization of the non-pecuniary
utility from staying at home.

.. math::
   :label: UtilityHome

   \zeta_5(k_{3,t}, h_t, t, e_{j,5}, \epsilon_{5,1}) =  e_{j,5} & +
   \gamma_{5,4} \cdot \mathbb{1}[18 \leq t \leq 20] + \gamma_{5,5}
   \cdot \mathbb{1}[t \geq 21] \\ \nonumber
   & +\vartheta_{1} \cdot \mathbb{1}[h_t \geq 12] + \vartheta_{2} \cdot
   \mathbb{1}[h_t \geq 16] +  \vartheta_3 \cdot \mathbb{1}[k_{3,t} = 1]  \\
   & + \epsilon_{5,t}

Staying at home as a young adult :math:`\gamma_{5, 4}` is less stigmatic as
doing so while already being an adult :math:`\gamma_{5,5}`. Additionally,
possessing a degree  :math:`(\vartheta_1, \vartheta_2)` or leaving the
military prematurely :math:`\vartheta_3` influences the immediate utility.


Wage component
--------------
The wage component :math:`w_{a}(\cdot)` for the working alternatives is given
by the product of the market-equilibrium rental price :math:`r_{a}` and an
occupation-specific skill level :math:`x_{a}(\cdot)`. The latter is determined
by the overall level of human capital.

.. math::

   w_{a}(\cdot) = r_{a} \, x_{a}(\cdot)

This specification leads to a standard logarithmic wage equation in which the c
onstant term is the skill rental price :math:`\ln(r_{a})` and wages follow a
log-normal distribution.

The occupation-specific skill level :math:`x_{a}(\cdot)` is determined by a
skill production function, which includes a deterministic component
:math:`\Gamma_a(\cdot)` and a multiplicative stochastic productivity shock
:math:`\epsilon_{a,t}`.

.. math::
   x_{a}(\boldsymbol{k_t}, h_t, t, a_{t-1}, e_{j, a}, \epsilon_{a,t}) = \exp
   \big( \Gamma_{a}(\boldsymbol{k_t},  h_t, t, a_{t-1}, e_{j,a}) \cdot
   \epsilon_{a,t} \big)


Blue-collar
^^^^^^^^^^^^
Equation :eq:`SkillLevelBlueCollar` shows the parameterization of the
deterministic component of the skill production function.

.. math::
   :label: SkillLevelBlueCollar

    \Gamma_1(\boldsymbol{k_t}, h_t, t, a_{t-1}, e_{j, 1}) = e_{j,1} & +
    \beta_{1,1} \cdot h_t + \beta_{1, 2} \cdot \mathbb{1}[h_t \geq 12] +
    \beta_{1,3} \cdot \mathbb{1}[h_t\geq 16]\\
    & + \gamma_{1, 1} \cdot  k_{1,t} + \gamma_{1,2} \cdot  (k_{1,t})^2 +
    \gamma_{1,3} \cdot  \mathbb{1}[k_{1,t} > 0] \\
    & + \gamma_{1,4} \cdot  t + \gamma_{1,5} \cdot \mathbb{1}[t < 18]\\
    & + \gamma_{1,6} \cdot \mathbb{1}[a_{t-1} = 1] + \gamma_{1,7} \cdot
    k_{2,t} + \gamma_{1,8} \cdot  k_{3,t}


There are several notable features. The first part of the skill production
function is motivated by Mincer (1958, :cite:`Mincer.1958`) and Mincer and
Polachek (1974, :cite:`Mincer.1974`) and hence linear in years of completed
schooling :math:`\beta_{1,1}`, quadratic in experience
(:math:`\gamma_{1,1}, \gamma_{1,2}`), and separable between the two of them.
There are so-called sheep-skin effects (Spence, 1973, :cite:`Spence.1973`,
Jaeger and Page, 1996, :cite:`Jaeger.1996`) associated with completing a high
school :math:`\beta_{1,2}` and graduate :math:`\beta_{1,3}` education that
capture the impact of completing a degree beyond just the associated
years of schooling. Also, skills depreciate when not employed in a
blue-collar occupation in the preceding period :math:`\gamma_{1,6}`.
Other work experience (:math:`\gamma_{1,7}, \gamma_{1,8}`) is transferable.

White-collar
^^^^^^^^^^^^
The wage component from working in a white-collar occupation is specified
analogously. Equation :eq:`SkillLevelWhiteCollar` shows the parameterization
of the deterministic component of the skill production function.

.. math::
   :label: SkillLevelWhiteCollar

    \Gamma_2(\boldsymbol{k_t}, h_t, t, a_{t-1}, e_{j,2}) = e_{j,2} & +
    \beta_{2,1} \cdot h_t + \beta_{2, 2} \cdot \mathbb{1}[h_t \geq 12] +
    \beta_{2,3} \cdot \mathbb{1}[h_t\geq 16] \\
    & + \gamma_{2, 1} \cdot  k_{2,t} + \gamma_{2,2} \cdot
    (k_{2,t})^2 + \gamma_{2,3} \cdot  \mathbb{1}[k_{2,t} > 0] \\
    & + \gamma_{2,4} \cdot  t + \gamma_{2,5} \cdot \mathbb{1}[t < 18] \\
    & + \gamma_{2,6} \cdot  \mathbb{1}[a_{t-1} = 2]  + \gamma_{2,7}
    \cdot  k_{1,t} + \gamma_{2,8} \cdot  k_{3,t}


Military
^^^^^^^^
Equation :eq:`SkillLevelMilitary` shows the parameterization of the
deterministic component of the skill production function.

.. math::
   :label: SkillLevelMilitary

   \Gamma_3( k_{3,t}, h_t, t, e_{j,3}) = e_{j,3} & + \beta_{3,1} \cdot h_t \\
   \nonumber &+ \gamma_{3,1} \cdot  k_{3,t} + \gamma_{3,2} \cdot (k_{3,t})^2
   + \gamma_{3,3} \cdot \mathbb{1}[k_{3,t} > 0]\\
   \nonumber& + \gamma_{3,4} \cdot t + \gamma_{3,5} \cdot \mathbb{1}[t < 18]

Contrary to the civilian sector there are no sheep-skin effects from
graduation (:math:`\beta_{3,2} = \beta_{3,3}= 0`). The previous occupational
choice has no influence (:math:`\gamma_{3,6}= 0`) and any experience other
than military is non-transferable (:math:`\gamma_{3,7} = \gamma_{3,8} = 0`).

**Remark**: Our parameterization for the immediate utility of serving in the
military differs from Keane and Wolpin (1997, :cite:`Keane.1997`) as we
remain unsure about their exact specification. The authors state in
Footnote 31 (p.498) that the constant for the non-pecuniary utility
:math:`\alpha_{3,t}` depends on age. However, we are unable to determine
the precise nature of the relationship. Equation (C3) (p.521) also indicates
no productivity shock :math:`\epsilon_{a,t}` in the wage component.
Table 7 (p.500) reports such estimates.

.. raw:: html

   <div
    <p class="d-flex flex-row gs-torefguide">
        <span class="badge badge-info">To Explanation</span></p>
    <p>The operationalization of the model allows to proceed with the calibration as
       described in<a href="calibration.html">
       Calibration</a> </p>
   </div>
