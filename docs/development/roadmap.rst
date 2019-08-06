.. _roadmap:

=======
Roadmap
=======

We aim for improvements to ``respy`` in Economics, Software Engineering,
and Numerical Methods.

Economics and Statistics
========================

All topics listed here can be tackled as part of a bachelor or master thesis. If you are
interested, please contact us!

Explore Simulation Based Estimation
-----------------------------------

We want to add simulation based estimation to ``respy`` and compare the accuracy of
parameters estimated with maximum likelihood and simulation based methods. As ``respy``
already has the ability to simulate data, it would be very simple to implement method of
simulated moments or indirect inference estimation. As part of this project, we could
also experiments with approaches that make the criterion function smooth and therefore
allow the use of fast optimizers. A starting point could be `Frazier, Oka and Zhu (2019)
<https://doi.org/10.1016/j.jeconom.2019.06.003>`_

Estimate a model by CCP
-----------------------

Implement estimation by CCP and outline the trade-offs and discuss the validity of
estimation. This could be combined with having a dataset simulated that does not conform
to the CCP model and the ``respy`` model.

Estimate a model by Approximate Bayesian Computation
----------------------------------------------------

Use the `ABCpy package <https://arxiv.org/pdf/1711.04694.pdf>`_ and ``respy``'s
simulation capabilities to estimate the model via approximate Bayesian computation.
Compare it against other estimation methods in terms of computational burden and
precision of the estimates.

Sparse Maximization and Human Capital Investment
------------------------------------------------

Gabaix (2014) proposes a fully tractable, unifying theory of limited attention in
decision-making. The idea is that the decision-maker pays less or no attention to some
features of the situation. A potential application of sparse maximization is human
capital investment, since young individuals could (partially or even fully) neglect some
relevant features, which could tilt their choices. This may imply that a considerable
share of the US labor force is miss-allocated.

For more information check out the :download:`full description
<../_static/thesis_proposals/Sparsity_and_human_capital.pdf>`

Salience Theory and Human Capital Investment
--------------------------------------------

Bordalo, Gennaioli and Shleifer (2013) propose a unifying theory of salience in
decision-making. An attribute is salient when it “stands out” relative to the
alternative choices. A potential application of salience theory is human capital
investment, since young individuals could attach disproportionately high attention to
professions with salient returns, which could tilt their choices.

For more information check out the :download:`full description
<../_static/thesis_proposals/Salience_and_human_capital.pdf>`


Numerical Methods
=================

Improve numerical integration
-----------------------------

We use numerical integration to calculate value functions and choice probabilities
in the maximum likelihood estimation. Currently we use a smoothed monte-carlo
integration for both. For a thesis a student could explore how the accurracy and
speed of the integrals changes with the following strategies:

- Use a GHK simulator instead of current monte-carlo integration
- Use gaussian quadrature for choice probabilities
- Allow for restrictions on the correlation structure of the shocks that make it
    possible to reduce the dimensionality of the integrals.

Starting points are the following papers:

- `Skrainka and Judd (2011) <https://dx.doi.org/10.2139/ssrn.1870703>`_
- `Dunnet (1989) <https://doi.org/10.2307/2347754>`


Benchmark Different Optimizers
------------------------------

Explore the speed and reliability of local and global optimizers for maximum likelihood
estimation of the model. The results should be transferable to other estimation problems
with a noisy criterion function. Most relevant optimizers should already be implemented
in estimagic. Otherwise they can be added easily.
