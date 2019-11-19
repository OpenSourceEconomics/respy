Changes
=======

This is a record of all past ``respy`` releases and what went into them in reverse
chronological order. We follow `semantic versioning <https://semver.org/>`_ and all
releases are available on `Anaconda.org
<https://anaconda.org/opensourceeconomics/respy>`_.

2.0.0 - 2019-
-------------

- `#177 <https://github.com/OpenSourceEconomics/respy/pull/177>`_ removes all Fortran
  files and ensures that all tests still run through.
- `#193 <https://github.com/OpenSourceEconomics/respy/pull/193>`_ continues on the
  removal of Fortran.
- `#199 <https://github.com/OpenSourceEconomics/respy/pull/199>`_ makes the reward
  components modular.
- `#200 <https://github.com/OpenSourceEconomics/respy/pull/200>`_ implements the Kalman
  filter which allows to estimate measurement error in wages.
- `#201 <https://github.com/OpenSourceEconomics/respy/pull/201>`_ implements a flexible
  state space which is first and foremost flexible in the number of choices with
  experience and wages, but open to be extended.
- `#204 <https://github.com/OpenSourceEconomics/respy/pull/204>`_ adds more thesis
  proposals.
- `#205 <https://github.com/OpenSourceEconomics/respy/pull/205>`_ implements Azure
  Pipelines as the major CI, but we still rely on Travis-CI for deploying the package to
  PyPI.
- `#206 <https://github.com/OpenSourceEconomics/respy/pull/206>`_ prepares estimation
  with `estimagic <https://github.com/OpenSourceEconomics/estimagic>`_
- `#208 <https://github.com/OpenSourceEconomics/respy/pull/208>`_ implements parsing of
  choices from data frame.
- `#209 <https://github.com/OpenSourceEconomics/respy/pull/209>`_ adds parameterizations
  of Keane and Wolpin (1997). Special thanks to `janosg <https://github.com/janosg>`_
  who implemented an adaptive rescaling step within the likelihood aggregation which
  prevents under- and overflows in the contributions.
- `#211 <https://github.com/OpenSourceEconomics/respy/pull/211>`_ generalizes the
  construction of type probabilities with arbitrary covariates.
- `#221 <https://github.com/OpenSourceEconomics/respy/pull/221>`_ implements a new
  interface for the simulation which is similar to the estimation and reduces runtime
  for multiple simulations by a factor of four.
- `#223 <https://github.com/OpenSourceEconomics/respy/pull/223>`_ logs rental prices in
  parameterizations. The parameters of wage components correspond to the ``ln(wage)``
  equation.
- `#236 <https://github.com/OpenSourceEconomics/respy/pull/236>`_ implements a periodic
  indexer.

1.2.1 - 2019-05-19
------------------

- `#170 <https://github.com/OpenSourceEconomics/respy/pull/170>`_ adds a test for
  inadmissible states in the state space.
- `#180 <https://github.com/OpenSourceEconomics/respy/pull/180>`_ adds a long
  description to the PYPi package.
- `#181 <https://github.com/OpenSourceEconomics/respy/pull/181>`_ implements `nbsphinx
  <https://nbsphinx.readthedocs.io/en/latest/>`_ for a documentation based on notebooks
  and reworks structure and graphics.
- `#183 <https://github.com/OpenSourceEconomics/respy/pull/183>`_ adds a small set of
  regression tests.
- `#185 <https://github.com/OpenSourceEconomics/respy/pull/185>`_ adds a list of topics
  for theses.
- `#186 <https://github.com/OpenSourceEconomics/respy/pull/186>`_ replaces statsmodels
  as a dependency with our own implementation.

1.2.0 - 2019-04-23
------------------

This is the last release with a Fortran implementation. Mirrors 1.2.0-rc.1.

1.2.0-rc.1 - 2019-04-23
-----------------------

- `#162 <https://github.com/OpenSourceEconomics/respy/pull/162>`_ is a wrapper around
  multiple PRs in which a new Python version is implemented.
- `#150 <https://github.com/OpenSourceEconomics/respy/pull/150>`_ implements a new
  interface.
- `#133 <https://github.com/OpenSourceEconomics/respy/pull/133>`_ and `#140
  <https://github.com/OpenSourceEconomics/respy/pull/140>`_ add Appveyor to test respy
  on Windows.

1.1.0 - 2018-03-02
------------------

- Undocumented release

1.0.0 - 2016-08-10
------------------

This is the initial release of the ``respy`` package.
