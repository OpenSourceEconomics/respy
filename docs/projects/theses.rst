Theses
======

Here is a list of Master's theses submitted by students of economics from the University
of Bonn.

----

Maokomatanda, L. (2020). Sensitivity Analysis for Structural Microeconomic Models using Shapley Values.

    The parametric uncertainties in the analysis of structural microeconometric models are
    ubiquitous. However, sensitivity analysis is rare and particularly challenging in this
    setting due to many correlated parameters. Shapely values with their foundation in game theory
    appear particularly suited in this case. This thesis computes the Shapely values for the
    estimated model parameters in Keane and Wolpin (1994) to assess their relative quantitative
    importance for the model's counterfactual predictions.

Contact: `@lindamaok899 <https://github.com/lindamaok899>`_

---

Mensinger, T. (2020). On the Use of Surrogate Models in Structural Microeconometrics.

    The thesis applies the ideas of surrogate modeling to the human capital model by Keane and
    Wolpin (1994) to facilitate the application of uncertainty propagation and sensitivity analysis.
    The analysis demonstrates that surrogate models do not outperform a brute force approach when
    propagating the model's parametric uncertainty to selected quantities of interest to
    economists. However, surrogate models are essential to also conduct a subsequent sensitivity
    analysis and machine learning methods are useful to calibrate a surrogate to high precision.

Contact: `@timmens <https://github.com/timmens>`_

----

Gehlen, A. (2020). Simulation-Based Estimation of Discrete Choice Dynamic Life-Cycle
Models.

    The thesis revisits the models by Keane and Wolpin (1994, 1997) to explore Method of
    Simulated Moments (MSM) estimation as an alternative to the Simulated Maximum
    Likelihood (SML) approach used by the authors. The thesis discusses the various
    calibration choices needed to construct an appropriate MSM criterion function for
    estimation, as well as the challenges that come with optimization of the criterion.
    The analysis demonstrates that the MSM can be effectively employed for model
    estimation but simultaneously shows that results are very sensitive to calibration
    choices.

Contact: `@amageh <https://github.com/amageh>`_

----

Suchy, R. (2020). Robust Human Capital Investment.

    The thesis enriches the treatment of uncertainty in discrete choice dynamic
    programming models applied to human capital investment decisions. It extends the
    Keane and Wolpin (1997) life cycle model by a deliberate account of ambiguity.
    Individuals fear model misspecification and seek robust decision rules that work
    well over a whole range of models about their economic environment. The
    literature on robust optimization informs the implementation of a robust value
    function iteration procedure that induces robust policies. The resulting robust
    human capital model is calibrated to a sample of young men from the National
    Longitudinal Survey. The empirical validation of the model testifies its
    capability to appropriately match behavioral patterns in the data, to predict
    occupational choice decisions, and to improve the out-of-sample fit compared to
    a risk-only model. It carries novel implications for the ex-ante evaluation of
    educational policies. In case of a tuition subsidy, predictions from a
    misspecified risk-only model may embellish their efficacy. Given a particular
    tuition subsidy, a targeted intervention that reduces the level of ambiguity may
    increase the demand for education.

Contact: `@rafaelsuchy <https://github.com/rafaelsuchy>`_

----

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

    The thesis lays the foundation for **respy** to become a general framework for the
    estimation of discrete choice dynamic programming models in labor economics. It
    showcases the ability to represent Keane and Wolpin (1994), Keane and Wolpin (1997),
    and Keane and Wolpin (2000) within a single implementation.

Contact: `@tobiasraabe <https://github.com/tobiasraabe>`_
