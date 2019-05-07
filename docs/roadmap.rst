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

Implement estimation by CCP and outline the tradeoffs and discuss the validity of
estimation. This could be combined with having a dataset simulated that does not conform
to the CCP model and the respy model.


Software Engineering
====================

* research the *hypothesis* package to replace the hand-crafted property-based testing
  routines


Numerical Methods
=================

* link the package to optimization toolkits such as *TAO* or *HOPSPACK*
* implement additional integration strategies following `Skrainka and Judd (2011)
  <https://dx.doi.org/10.2139/ssrn.1870703>`_
