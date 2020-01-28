Changes
=======

This is a record of all past ``respy`` releases and what went into them in reverse
chronological order. We follow `semantic versioning <https://semver.org/>`_ and all
releases are available on `Anaconda.org
<https://anaconda.org/opensourceeconomics/respy>`_.

2.0.0 - 2019-
-------------

- :gh:`177` removes all Fortran files and ensures that all tests still run through.
- :gh:`193` continues on the removal of Fortran.
- :gh:`199` makes the reward components modular (:ghuser:`janosg`).
- :gh:`200` implements the Kalman filter which allows to estimate measurement error in
  wages (:ghuser:`janosg`).
- :gh:`201` implements a flexible state space which is first and foremost flexible in
  the number of choices with experience and wages, but open to be extended.
- :gh:`204` adds more thesis proposals.
- :gh:`205` implements Azure Pipelines as the major CI, but we still rely on Travis-CI
  for deploying the package to PyPI (:ghuser:`tobiasraabe`).
- :gh:`206` prepares estimation with `estimagic
  <https://github.com/OpenSourceEconomics/estimagic>`_
- :gh:`208` implements parsing of choices from data frame.
- :gh:`209` adds parameterizations of Keane and Wolpin (1997) plus adaptive rescaling
  step within the likelihood aggregation which prevents under- and overflows in the
  contributions (:ghuser:`janosg`).
- :gh:`211` generalizes the construction of type probabilities with arbitrary
  covariates.
- :gh:`221` implements a new interface for the simulation which is similar to the
  estimation and reduces runtime for multiple simulations by a factor of four.
- :gh:`236` implements a periodic indexer.
- :gh:`240` makes previous choices in the state space optional.
- :gh:`245` create continuation values dynamically from value functions.
- :gh:`256` implements quasi-random low discrepancy sequences for better Monte-Carlo
  integrations (:ghuser:`rafaelsuchy`).
- :gh:`262` moves the distribution of previous choices, initial and maximum experience
  to params.
- :gh:`268` makes seeding in ``respy`` a lot more robust by using incrementally
  increasing sequences.
- :gh:`269` improves the generation of conditional draws with and without measurement
  error in the likelihood calculation (:ghuser:`janosg`).
- :gh:`275` fixes a bug in calculation of wage probabilities.
- :gh:`277` adds three different simulation methods: n-step-ahead simulation with
  sampling or data and one-step-ahead simulation.
- :gh:`278`, :gh:`279`, and :gh:`280` implement three functions, log softmax, softmax,
  and logsumexp, which reduce the likelihood of under- and overflows and save
  information (:ghuser:`tobiasraabe`).
- :gh:`282` adds an interface for the estimation of models with the method of simulated
  moments (:ghuser:`amageh`, :ghuser:`momo2561057`, :ghuser:`tobiasraabe`).
- :gh:`285` adds the ability to generate a set of constraint for example models.
- :gh:`288` fixes an error in the simulation of choice probabilities introduced by
  :gh:`278` (:ghuser:`peisenha`).
- :gh:`296` contributes a new toy model to respy: The Robinson Crusoe Economy
  (:ghuser:`tobiasraabe`, :ghuser:`peisenha`)
- :gh:`299` adds the information on the previous choice to individuals at age sixteen to
  the original data from Keane and Wolpin (1997). Special thanks to :ghuser:`bekauf` for
  the data preparation.
- :gh:`300` aligns respy functions with the new data in :gh:`299`.
- :gh:`314` fixes two parameters in KW97 and KW2000 (:ghuser:`tostenzel`,
  :ghuser:`tobiasraabe`).
- :gh:`316` changes the invalid index value for the indexer to prevent silent errors
  (:ghuser:`tobiasraabe`).
- :gh:`319` adds a page for projects using ``respy`` (:ghuser:`tobiasraabe`).


1.2.1 - 2019-05-19
------------------

- :gh:`170` adds a test for inadmissible states in the state space.
- :gh:`180` adds a long description to the PYPi package.
- :gh:`181` implements `nbsphinx <https://nbsphinx.readthedocs.io/en/latest/>`_ for a
  documentation based on notebooks and reworks structure and graphics.
- :gh:`183` adds a small set of regression tests.
- :gh:`185` adds a list of topics for theses.
- :gh:`186` replaces ``statsmodels`` as a dependency with our own OLS implementation.

1.2.0 - 2019-04-23
------------------

This is the last release with a Fortran implementation. Mirrors 1.2.0-rc.1.

1.2.0-rc.1 - 2019-04-23
-----------------------

- :gh:`162` is a wrapper around multiple PRs in which a new Python version is
  implemented.
- :gh:`150` implements a new interface.
- :gh:`133` and :gh:`140` add Appveyor to test respy on Windows.

1.1.0 - 2018-03-02
------------------

- Undocumented release.

1.0.0 - 2016-08-10
------------------

This is the initial release of the ``respy`` package.
