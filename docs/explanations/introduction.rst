.. _introduction:

Introduction
==============

``respy`` is an open source framework written in Python for the simulation and
estimation of some finite-horizon discrete choice dynamic programming (DCDP) models.
In comparison to simple reduced-form analysis, these models allow the estimation
of structural parameters which reflect agents' preferences and beliefs by assuming
that agents are forward-looking and maximize expected intertemporal payoffs.
Over the last decades, finite-horizon DCDP models have become a popular tool to
answer research questions in areas of economics such as
labor economics, industrial organization, economic demography, health economics,
development economics, political economy and marketing.

As is common with structural economic models, finite-horizon DCDP models oftentimes
rely on strong assumptions regarding unobservable state variables and error terms.
``respy`` focuses on the estimation of so-called **Eckstein-Keane-Wolpin (EKW) models**.
In accordance with Aguirregabiria and Mira (2010, :cite:`Aguirregabiria.2010`)
, we classify a DCDP model as an EKW model if it departures from standard
DCDP modeling by relaxing at least one of the following assumptions:

1. The one-period utility function does not have to be *additively separable* in
   its observable and unobservable components but can instead feature different
   compositions, e. g. multiplicative separability.

2. Observable payoff variables can be *choice-censored* and the value of the payoff
   variable does not have to be independent of the error term :math:`\epsilon`,
   conditional on the values of the decision and observable state variables.

3. *Permanent unobserved heterogeneity* is allowed to exist, i. e. the unobserved
   state variables do not have to be independently and identically distributed
   over agent and over time. As an example, the seminal work of Keane and Wolpin
   (1997, :cite:`Keane.1997`) introduces permanent unobserved heterogeneity by
   assigning each individual to one of four type.

4. Unobservables may be *correlated across choice alternatives*, i. e. unobserved
   state variables do not have to be independent across alternatives.

.. raw:: html

   <div
    <p class="d-flex flex-row gs-torefguide">
        <span class="badge badge-info">To recommended readings</span></p>
    <p>To learn more about DCDP models and related topics, check out
       <a href="recommended_reading.html">
       Recommended Reading</a> </p>
   </div>
