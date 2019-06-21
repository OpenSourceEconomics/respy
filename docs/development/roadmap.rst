.. _roadmap:

=======
Roadmap
=======

We aim for improvements to ``respy`` in three domains: Economics, Software Engineering,
and Numerical Methods.


Economics and Statistics
========================

All topics listed here can be tackled as part of a bachelor or master thesis. If you are
interested, please contact us!

Support Keane and Wolpin (1997)
-------------------------------

We want to support the full model of `Keane and Wolpin (1997)
<https://doi.org/10.1086/262080>`_. This is almost completed. We just need to implement
a fifth choice (the military sector).

Explore Simulation Based Estimation
-----------------------------------

We want to add simulation based estimation to respy and compare the accuracy of
parameters estimated with maximum likelihood and simulation based methods. As respy
already has the ability to simulate data, it would be very simple to implement method of
simulated moments or indirect inference estimation.

Benchmark Different Optimizers
------------------------------

Explore the speed and reliability of local and global optimizers for maximum likelihood
estimation of the model. The results should be transferable to other estimation problems
with a noisy criterion function.

Improve the Likelihood Estimation
---------------------------------

We suspect that the generation of multivariate normal shocks in KW94 is flawed. We want
to derive a numerically robust method for correct sampling from a multivariate normal
distribution. Compare the precision of estimates with the old and new methods on
simulated samples. This is especially interesting for students who want to deeply
understand estimation of structural models with maximum likelihood.

Estimate a model by CCP
-----------------------

Implement estimation by CCP and outline the trade-offs and discuss the validity of
estimation. This could be combined with having a dataset simulated that does not conform
to the CCP model and the respy model.


Estimate a model by Approximate Bayesian Computation
----------------------------------------------------

Use the `ABCpy package <https://arxiv.org/pdf/1711.04694.pdf>`_ and respy's simulation capabilities to estimate the model via approximate bayesian computation. Compare it against other estimation methods in terms of computational burden and precision of the estimates.


Sparse Maximization and Human Capital Investment
------------------------------------------------

Gabaix (2014) proposes a fully tractable, unifying theory of
limited attention in decision-making. The idea is that the
decision-maker pays less or no attention to some features of the
situation. A potential application of sparse maximization is human
capital investment, since young individuals could (partially
or even fully) neglect some relevant features, which could tilt
their choices. This may imply that a considerable share of the
US labor force is misallocated.

Salience Theory and Human Capital Investment
--------------------------------------------

Bordalo, Gennaioli and Shleifer (2013) propose a unifying theory of
salience in decisionmaking. An attribute is salient when it “stands
out” relative to the alternative choices. A potential application of
salience theory is human capital investment, since young individuals
could attach disproportionately high attention to professions with
salient returns, which could tilt their choices.



Software Engineering
====================

* research the *hypothesis* package to replace the hand-crafted property-based testing
  routines


Numerical Methods
=================

* link the package to optimization toolkits such as *TAO* or *HOPSPACK*
* implement additional integration strategies following `Skrainka and Judd (2011)
  <https://dx.doi.org/10.2139/ssrn.1870703>`_
