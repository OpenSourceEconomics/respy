.. _roadmap:

=======
Roadmap
=======

We aim for improvements to ``respy`` in Economics, Statistics, and Numerical Methods.

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

CCP and the Estimation of Nonseparable Dynamic Modes
----------------------------------------------------

In a recent paper `Kristensen, Nesheim, and de Paulo (2015)
<https://www.ucl.ac.uk/~uctpand/hotzmiller-2015-11-21.pdf>`_ generalize the conditional
choice probabilities (CCP) estimator (`Hotz, Miller (1993)
<https://www.jstor.org/stable/2298122?>`_) to non-separable economic models. However,
they are still missing an empirical application of their method as a proof of concept.
The ``respy`` package offers a suitable starting point.

Estimate a model by Approximate Bayesian Computation (ABC)
----------------------------------------------------------

Use the `ABCpy package <https://arxiv.org/pdf/1711.04694.pdf>`_ and ``respy``'s
simulation capabilities to estimate the model via ABC. Compare it against other
estimation methods in terms of computational burden and precision of the estimates.

Sparse Maximization and Human Capital Investment
------------------------------------------------

Gabaix (2014) proposes a fully tractable, unifying theory of limited attention in
decision-making. The idea is that the decision-maker pays less or no attention to some
features of the situation. A potential application of sparse maximization is human
capital investment, since young individuals could (partially or even fully) neglect some
relevant features, which could tilt their choices. This may imply that a considerable
share of the US labor force is miss-allocated. For more information check out the
:download:`full description
<../_static/thesis_proposals/Sparsity_and_human_capital.pdf>`

Salience Theory and Human Capital Investment
--------------------------------------------

Bordalo, Gennaioli and Shleifer (2013) propose a unifying theory of salience in
decision-making. An attribute is salient when it “stands out” relative to the
alternative choices. A potential application of salience theory is human capital
investment, since young individuals could attach disproportionately high attention to
professions with salient returns, which could tilt their choices. For more information
check out the :download:`full description
<../_static/thesis_proposals/Salience_and_human_capital.pdf>`

Numerical Methods
=================

Improve numerical integration
-----------------------------

We use numerical integration to calculate value functions and choice probabilities in
the maximum likelihood estimation. Currently we use a smoothed Monte-Carlo integration
for both. For a thesis a student could explore how the accuracy and speed of the
integrals changes with the following strategies:

- Use a GHK simulator instead of current Monte-Carlo integration.
- Use Gaussian quadrature for choice probabilities.
- Allow for restrictions on the correlation structure of the shocks that make it
  possible to reduce the dimensionality of the integrals.

Starting points are the following papers:

- `Skrainka and Judd (2011) <https://dx.doi.org/10.2139/ssrn.1870703>`_
- `Dunnet (1989) <https://doi.org/10.2307/2347754>`_

Benchmark Different Optimizers
------------------------------

Explore the speed and reliability of local and global optimizers for maximum likelihood
estimation of the model. The results should be transferable to other estimation problems
with a noisy criterion function. Most relevant optimizers should already be implemented
in ``estimagic``. Otherwise they can be added easily.

Approximate Dynamic Programming (ADP)
-------------------------------------

We want to explore the usefulness of ADP techniques for solving large scale structural
economic models. The seminal references is `Powell (2011)
<http://adp.princeton.edu/>`_.
