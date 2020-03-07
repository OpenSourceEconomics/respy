Projects
========

``respy`` is actively used as a research tool. Here is a list of projects to showcase
the capabilities and kind of questions which are tackled with ``respy``.

If you want to feature your project in this list, open an `issue or pr
<https://github.com/OpenSourceEconomics/respy>`_ and submit a title, a summary, links to
further resources, and contact details.

Research
--------

Bhuller, M., Eisenhauer, P. and Mendel, M. (2020). The Option Value of Education.
*Working Paper*.

    We provide a comprehensive account of the returns to education using Norwegian
    population panel data with nearly career-long earnings histories. We use variation
    induced by a mandatory schooling reform for an instrumental variables strategy as
    well as the validation of a full structural model. We discuss the trade-offs between
    the two approaches. Using the structural model, we go beyond the standard return
    concepts such as Mincer returns and the internal rate of return. This allows us to
    account for the sequential resolution of uncertainty and nonlinearities in the
    returns to education. Both give rise to option values as each additional year of
    schooling provides information about the value of different schooling choices and
    new opportunities become available. We are thus able to estimate the true return to
    education and find an important role for option values.

Contact: `@peisenha <https://github.com/peisenha>`_, `@mo2561057
<https://github.com/mo2561057>`_

----

Eisenhauer, P. and Suchy, R. (2020). Robust Human Capital Investment under Risk and
Ambiguity. *Working Paper*.

    We build on the prototypical life cycle model of human capital investment Keane and
    Wolpin (1994) and study individual decision-making under risk as well as ambiguity.
    Individuals fear model misspecification and seek robust decisions that work well
    over a whole range of models about their economic environment. We describe the
    individual's decision problem as a robust Markov decision process. Our Monte Carlo
    analysis indicates that the empirical finding of large psychic cost of schooling is
    in part due to model misspecification by econometricians who only analyze individual
    investment decisions under risk. This changes the mechanisms driving schooling
    decisions and affects the ex ante evaluation of tuition policies.

Contact: `@peisenha <https://github.com/peisenha>`_, `@rafaelsuchy
<https://github.com/rafaelsuchy>`_

----

Eisenhauer, P. (2019). `The Approximate Solution of Finite-Horizon Discrete Choice
Dynamic Programming Models: Revisiting Keane & Wolpin (1994)
<https://doi.org/10.1002/jae.2648>`_. *Journal of Applied Econometrics, 34* (1),
149-154.

    The estimation of finite‐horizon discrete‐choice dynamic programming (DCDP) models
    is computationally expensive. This limits their realism and impedes verification and
    validation efforts. `Keane and Wolpin (Review of Economics and Statistics, 1994,
    76(4), 648–672) <https://doi.org/10.2307/2109768>`_ propose an interpolation method
    that ameliorates the computational burden but introduces approximation error. I
    describe their approach in detail, successfully recompute their original quality
    diagnostics, and provide some additional insights that underscore the trade‐off
    between computation time and the accuracy of estimation results.

Contact: `@peisenha <https://github.com/peisenha>`_


Thesis
------

Stenzel, T. (2020). Uncertainty Quantification for an Eckstein-Keane-Wolpin model with
correlated input parameters.

    The thesis analyzes the uncertainty of the effect of a 500 USD subsidy on annual
    tuition costs for higher education on the average years of education caused by the
    parametric uncertainty in the model of occupational choice by Keane and Wolpin
    (1994). This model output is called a quantity of interest (QoI). The uncertainty
    quantification (UQ) has two stages. The first stage is an uncertainty analysis, and
    the second stage is a quantitative global sensitivity analysis (GSA). The
    uncertainty analysis finds that the tuition subsidy has a mean effect of an increase
    of 1.5 years and a standard deviation of 0.1 years in education. For the qualitative
    GSA, I develop redesigned Elementary Effects based on Ge and Menendez (2017) for a
    model with correlated input parameters. Based on these Elementary Effects, I compute
    multiple aggregate statistics to quantify the impact of the uncertainty in one
    parameter on uncertainty in the QoI. However, the analysis does not lead to clear
    results as there is no consensus about how to interpret the aggregate statistics in
    this context - even for uncorrelated parameters.


Contact: `@tostenzel <https://github.com/tostenzel>`_

----

Massner, P. (2019). Modeling Wage Uncertainty in Dynamic Life-cycle Models.

    The thesis sheds light on the validity of the modeling assumptions of wage
    uncertainty in dynamic life-cycle models of human capital investment. Since the
    pioneering work of Keane and Wolpin (1997), a majority of studies in this field
    followed the approach of assuming serially independent productivity shocks to wages.
    In the case of Keane and Wolpin (1997), the independence assumption indeed
    simplifies the numerical solution of the model compared to more complex
    specifications, such as serial correlation. However, the assumption of i.i.d.
    productivity shocks seems to be quite narrow in light of findings of the
    reduced-form literature stream on wage dynamics.

Contact: `@PatriziaMassner <https://github.com/PatriziaMassner>`_

----

Raabe, T. (2019). A unified estimation framework for some discrete choice dynamic
programming models.

    The thesis lays the foundation for ``respy`` to become a general framework for the
    estimation of discrete choice dynamic programming models in labor economics. It
    showcases the ability to represent Keane and Wolpin (1994), Keane and Wolpin (1997),
    and Keane and Wolpin (2000) within a single implementation.

Contact: `@tobiasraabe <https://github.com/tobiasraabe>`_
