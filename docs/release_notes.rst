Release notes
=============

This is a record of all past **respy** releases and what went into them in reverse
chronological order. We follow `semantic versioning <https://semver.org/>`_ and all
releases are available on `Anaconda.org
<https://anaconda.org/opensourceeconomics/respy>`_.

2.1.0 - 2020-
-------------
- :gh:`381` Implements exogenous processes (:ghuser:`MaxBlesch`, :ghuser:`mo2561057`).
- :gh:`383` Fixes simulation with data and adds tests (:ghuser:`janosg`).
- :gh:`387` Fixes issue in documenation build (:ghuser:`amageh`).
- :gh:`389` Adjust criterion functions to return a scalar value or dictionary with
  additional information (:ghuser:`amageh`).
- :gh:`391` Adds how-to guide on example models and specifying a model (:ghuser:`amageh`;
  model specification guide draws on work by :ghuser:`tobiasraabe`
  and :ghuser:`rafaelsuchy`). Fixes bug in model validation for shock parameters.
- :gh:`395` Adds guides and tutorials for exogenous proccesses, covariates, and maximum 
  likelihood estimation. Improves structure and appearance of documentation. 
  (:ghuser:`MaxBlesch`, :ghuser:`amageh`).


2.0.0 - 2019-2020
-----------------

- :gh:`177` removes all Fortran files and ensures that all tests still run through
  (:ghuser:`tobiasraabe`).
- :gh:`193` continues on the removal of Fortran (:ghuser:`tobiasraabe`).
- :gh:`199` makes the reward components modular (:ghuser:`janosg`).
- :gh:`200` implements the Kalman filter which allows to estimate measurement error in
  wages (:ghuser:`janosg`).
- :gh:`201` implements a flexible state space which is first and foremost flexible in
  the number of choices with experience and wages, but open to be extended
  (:ghuser:`tobiasraabe`).
- :gh:`204` adds more thesis proposals (:ghuser:`tobiasraabe`).
- :gh:`205` implements Azure Pipelines as the major CI, but we still rely on Travis-CI
  for deploying the package to PyPI (:ghuser:`tobiasraabe`).
- :gh:`206` prepares estimation with `estimagic
  <https://github.com/OpenSourceEconomics/estimagic>`_ (:ghuser:`tobiasraabe`).
- :gh:`208` implements parsing of choices from DataFrame (:ghuser:`tobiasraabe`).
- :gh:`209` adds parameterizations of Keane and Wolpin (1997) plus adaptive rescaling
  step within the likelihood aggregation which prevents under- and overflows in the
  contributions (:ghuser:`janosg`).
- :gh:`211` generalizes the construction of type probabilities with arbitrary
  covariates.
- :gh:`221` implements a new interface for the simulation which is similar to the
  estimation and reduces runtime for multiple simulations by a factor of four
  (:ghuser:`tobiasraabe`).
- :gh:`230` allows the model to include observed variables which are time-invariant
  (:ghuser:`mo2561057`, :ghuser:`tobiasraabe`)
- :gh:`236` implements a periodic indexer (:ghuser:`tobiasraabe`).
- :gh:`240` makes previous choices in the state space optional (:ghuser:`tobiasraabe`).
- :gh:`245` create continuation values dynamically from value functions
  (:ghuser:`tobiasraabe`).
- :gh:`256` implements quasi-random low discrepancy sequences for better Monte-Carlo
  integrations (:ghuser:`rafaelsuchy`).
- :gh:`262` moves the distribution of previous choices, initial and maximum experience
  to params (:ghuser:`tobiasraabe`).
- :gh:`268` makes seeding in **respy** a lot more robust by using incrementally
  increasing sequences (:ghuser:`janosg`, :ghuser:`tobiasraabe`).
- :gh:`269` improves the generation of conditional draws with and without measurement
  error in the likelihood calculation (:ghuser:`janosg`).
- :gh:`275` fixes a bug in calculation of wage probabilities (:ghuser:`tobiasraabe`).
- :gh:`277` adds three different simulation methods: n-step-ahead simulation with
  sampling or data and one-step-ahead simulation (:ghuser:`tobiasraabe`).
- :gh:`278`, :gh:`279`, and :gh:`280` implement three functions, log softmax, softmax,
  and logsumexp, which reduce the likelihood of under- and overflows and save
  information (:ghuser:`tobiasraabe`).
- :gh:`282` adds an interface for the estimation of models with the method of simulated
  moments (:ghuser:`amageh`, :ghuser:`mo2561057`, :ghuser:`tobiasraabe`).
- :gh:`285` adds the ability to generate a set of constraint for example models
  (:ghuser:`tobiasraabe`).
- :gh:`288` fixes an error in the simulation of choice probabilities introduced by
  :gh:`278` (:ghuser:`peisenha`).
- :gh:`296` contributes a new toy model to respy: The Robinson Crusoe Economy
  (:ghuser:`tobiasraabe`, :ghuser:`peisenha`)
- :gh:`299` adds the information on the previous choice to individuals at age sixteen to
  the original data from Keane and Wolpin (1997). Special thanks to :ghuser:`bekauf` for
  the data preparation.
- :gh:`300` aligns respy functions with the new data in :gh:`299`
  (:ghuser:`tobiasraabe`).
- :gh:`310` introduces the separation between a core state space and dense dimensions of
  the state space which reduces memory consumption by a lot and makes respy scalable.
  :gh:`312` and :gh:`313` include changes to the simulation or maximum likelihood
  estimation which pave the way for :gh:`310`. (:ghuser:`tobiasraabe`)
- :gh:`314` fixes two parameters in KW97 and KW2000 (:ghuser:`tostenzel`,
  :ghuser:`tobiasraabe`).
- :gh:`316` changes the invalid index value for the indexer to prevent silent errors
  (:ghuser:`tobiasraabe`).
- :gh:`319` adds a page for projects using **respy** (:ghuser:`tobiasraabe`). :gh:`321`
  adds more projects.
- :gh:`320` adds ``add_noise_to_params()`` and makes the test suite faster, tests more
  random, moved to Github Actions, and more badges.
- :gh:`323` adds an informative message if simulated individuals cannot be mapped to
  states in the state space (:ghuser:`mo2561057`, :ghuser:`tobiasraabe`).
- :gh:`325` adds an how-to guide on numerical integration techniques
  (:ghuser:`rafaelsuchy`).
- :gh:`331` better parsing for observables (and exogenous processes) and better model
  tests and docstrings (:ghuser:`tobiasraabe`).
- :gh:`342` partitions the state space to parts which are specific to the period, choice
  set and dense values (:ghuser:`mo2561057`, :ghuser:`tobiasraabe`).
- :gh:`344` redesigns the documentation. The foundation are four categories of
  documents, tutorials, explanations, how-to guides, and reference guides
  (:ghuser:`tobiasraabe`).
- :gh:`347` allows to specify models with hyperbolic discounting
  (:ghuser:`SofiaBadini`).
- :gh:`356` adds how-to guide for estimation of parameters with msm (:ghuser:`amageh`).
- :gh:`357` adds a tutorial explaining the basic interface (:ghuser:`SofiaBadini`).
- :gh:`359` fixes a Numba deprecation warning and some errors in the constraints of
  pre-defined models.
- :gh:`361` adds standard deviations of parameters for example models
  (:ghuser:`timmens`).
- :gh:`363` enables msm function to return simulated moments or comparison plot data for
  use with `estimagic <https://github.com/OpenSourceEconomics/estimagic>`_
  (:ghuser:`amageh`).
- :gh:`366` adds comprehensive, exemplary explanation of model from 
  Keane and Wolpin (1997) to documentation (:ghuser:`bekauf`, :ghuser:`rafaelsuchy`). 
- :gh:`369` adds second set of parameters for kw_97 models (:ghuser:`amageh`).
- :gh:`371` changes the names of the criterion functions for maximum likelihood and msm
  estimation. Makes replacement functions optional for estimation with
  msm and sets identity matrix as default weighting matrix (:ghuser:`amageh`).
- :gh:`373` refactors the law of motion and simplifies the collection of child indices
  (:ghuser:`tobiasraabe`).
- :gh:`374` renames caching options to ``"cache_path"`` and ``"cache_compression"``
  (:ghuser:`tobiasraabe`).

*Releases prior to the second version were published on PyPI, but later deleted. You can
still checkout the following releases using the corresponding tags in the repository.*

1.2.1 - 2019-05-19
------------------

- :gh:`170` adds a test for inadmissible states in the state space.
- :gh:`180` adds a long description to the PyPI package.
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

This is the initial release of the **respy** package.
