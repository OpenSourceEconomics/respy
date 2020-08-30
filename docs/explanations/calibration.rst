.. _calibration:

Calibration Procedure
=====================

.. role:: boldblue

EKW models are calibrated to data on observed individual decisions and
experiences, given the assumption that individuals behave according to the
behavioral model. For example, the Keane and Wolpin (1997, :cite:`Keane.1997`)
model is calibrated  to a subsample of young men from the National Longitudinal
Survey 1979 (NLSY79). The :boldblue:`goal` is to back out information on
utility functions, preference parameters, and transition probabilities.
This requires to fully parameterize the model. We denote :math:`\theta`
the vector of structural parameters out of an admissible parameter space
:math:`\Theta`.

--------------------------------------------------------------------------------

.. rst-class:: centerblue

   The discount factor is pre-defined in ``respy`` and mandatory.
   It is located in params data frame under the key ``delta``.

--------------------------------------------------------------------------------

Under the paradigm of :boldblue:`revealed preferences` (Samuelson, 1938,
:cite:`Samuelson.1938`) structural preference parameters are estimable
with microdata on individual decisions. Generally, we have access to information
for :math:`i = 1, \dots, N` individuals at each point in time :math:`1, \dots, T_i`.
For each observation :math:`(i,t)` in the data we observe that action taken
:math:`a_{it}` some components of the utility :math:`\bar{u}_{it}`, and the
observable state space :math:`\bar{s}_t`. At time :math:`t` both the individual
and we as a researcher observe :math:`\bar{s}_t` but the stochastic component
:math:`\epsilon_t` is only observed by the individual. [#]_ In summary,
the :boldblue:`data structure` is given by

.. math::
   :label: data_structure

   \mathcal{D} = \big\{ a_{it}, \bar{s}_{it}, \bar{u}_{it}: i = 1, \dots, N;
   t = 1, \dots, T_i \big\}.

Given the present data structure numerous calibraion procedures exist (Davidson
\& MacKinnon, 2003, :cite:`Davidson.2003`; Gourieroux \& Monfort, 1996,
:cite:`Gourieroux.1996`). We will outline likelihood-based and simulation-based
calibration. Independent of the calibration criterion, it is necessary to solve
for the optimal policy :math:`\pi^*` at each candidate parameterization of the
model.

--------------------------------------------------------------------------------

.. rst-class:: centerblue

  ``respy`` supports the calibration via simulated maximum-likelihood and the
  method of simulated moments. Both can be called with ``respy.likelihood`` and
  ``respy.method_of_simulated_moments``, respectively.

--------------------------------------------------------------------------------

:boldblue:`Likelihood-based calibration` seeks to find the parameterization
:math:`\theta` that maximizes the likelihood function
:math:`\mathcal{L}(\theta \,|\, \mathcal{D})`, i.e. the probability of observing
the given data as a function of :math:`\theta`. As we only observe a subset
:math:`\bar{s}_{it}` of the state, we can determine the probability
:math:`p_{it}(a_{it}, \bar{u}_{it} \,|\, \bar{s}_{it}, \theta)` of individual
:math:`i` at time :math:`t` choosing :math:`a_{it}` and receiving :math:`u_{it}`
given a parametric assumption about the distribution of :math:`\epsilon_{it}`. [#]_
The objective function takes the following form:

.. math::
   :label: eq:likelihood_function

   \hat{\theta} \equiv \underset{\theta \in \Theta}{{\arg \max}}
   \underbrace{\prod_{i=1}^N \prod_{t = 1}^{T_i}
   p_{it}(a_{it}, \bar{u}_{it} \,|\, \bar{s}_{it}, \theta)}_{\mathcal{L}
   (\theta \,|\, \mathcal{D})}.

--------------------------------------------------------------------------------

.. rst-class:: centerblue

  The implementation in ``respy`` minimizes the
  simulated negative log-likelihood of the observed sample.

--------------------------------------------------------------------------------

:boldblue:`Simulation-based calibration` seeks to find the parameterization
:math:`\hat{\theta}` that yields a simulated data set :math:`M_S(\theta)` from
the model that closest resembles the observed data. More precisely, the goal is
to minimize the weighted squared distance between a set of moments
:math:`M_{\mathcal{D}}` computed on the observed data and the same set of
moments computed on the simulated data :math:`M_{\mathcal{S}}(\theta)`. The
objective function takes the following form:

.. math::
   :label: eq:likelihood_function

   \hat{\theta} \equiv \underset{\theta \in \Theta}{{\arg \min}}
   \big( M_{\mathcal{D}} - M_{\mathcal{S}}(\theta) \big)' \,
   W \, \big( M_{\mathcal{D}} - M_{\mathcal{S}}(\theta) \big).


The work by Eisenhauer, Heckman, and Mosso (2015, :cite:`Eisenhauer.2015`)
:boldblue:`compares the performance` of the MSM estimator against a standard
maximum likelihood estimator in a simplified dynamic discrete choice model of
schooling. Different to Keane and Wolpin (1994, :cite:`Keane.1994`; 1997,
:cite:`Keane.1997`) their restriction to binary choices of agents allows to
solve for the likelihood analytically and so dispenses the need for simulation
or interpolation. Their maximum likelihood estimates are close to the ''true''
structural objects of interest while MSM fails to recover some of them. At p.351
the authors provide a comparison of alternative weighting matrices.

--------------------------------------------------------------------------------

.. rst-class:: centerblue

   The implementation of MSM estimation in ``respy`` is extensively
   described in the tutorial on `Methods of Simulated Moments (MSM)
   <https://respy.readthedocs.io/en/latest/how_to_guides/msm.html>`_
   and the tutorial on `How to Estimate Model Parameters with MSM
   <https://respy.readthedocs.io/en/latest/how_to_
   guides/msm_estimation_exercise.html>`_.

--------------------------------------------------------------------------------

We have explained the economic model, its solution, one particular specification,
and the calibration procedure. The `Robinson Crusoe tutorial <https://
respy.readthedocs.io/en/latest/tutorials/robinson_crusoe.html>`_
provides a great applied resource to familiarize with the main functionalities
of ``respy``. Reading through will help you to set-up and calibrate your own
DCDP model.


.. rubric:: Footnotes

.. [#] The observable state space :math:`s_{it}` summarizes years of
       completed schooling, work experience, and choices.

.. [#] Notably, each different parameterization induces also a different
       probability distribution over the sequence of observed choices.
