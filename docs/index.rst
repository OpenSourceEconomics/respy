.. Keep the following section in sync with README.rst.

respy
=====

.. image:: https://anaconda.org/opensourceeconomics/respy/badges/version.svg
    :target: https://anaconda.org/OpenSourceEconomics/respy

.. image:: https://anaconda.org/opensourceeconomics/respy/badges/platforms.svg
    :target: https://anaconda.org/OpenSourceEconomics/respy

.. image:: https://readthedocs.org/projects/respy/badge/?version=latest
    :target: https://respy.readthedocs.io/en/latest

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT

.. image:: https://github.com/OpenSourceEconomics/respy/workflows/Continuous%20Integration%20Workflow/badge.svg?branch=master
    :target: https://github.com/OpenSourceEconomics/respy/actions?query=branch%3Amaster

.. image:: https://codecov.io/gh/OpenSourceEconomics/respy/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/OpenSourceEconomics/respy

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

----

**respy** is an open source framework written in Python for the simulation and
estimation of some finite-horizon discrete choice dynamic programming models. The group
of models which can be currently represented in **respy** are called
Eckstein-Keane-Wolpin models (Aguirregabiria and Mira (2010))

What makes **respy** powerful is that it allows to build and solve structural models in
weeks or months whose development previously took years. The design of **respy** allows
the researcher to flexibly add the following components to her model.

- **Any number of discrete choices** (e.g., working alternatives, schooling, home
  production, retirement) where each choice may yield a wage, may allow for experience
  accumulation and can be constrained by time, a maximum amount of accumulated
  experience or other characteristics.

- Condition the decision of individuals on its **previous choices** or their labor
  market history.

- Adding a **finite mixture** with any number of subgroups to account for unobserved
  heterogeneity among individuals as developed by Keane and Wolpin (1997).

- **Any number of time-constant observed state variables** (e.g., ability measures
  (Bhuller et al. (2020)), race (Keane and Wolpin (2000)), demographic variables) found
  in the data.

- Correct the estimation for **measurement error** in wages, either using a Kalman
  filter in maximum likelihood estimation or by adding the measurement error in
  simulation based approaches.

.. End of section

**respy**'s documentation is structured in four parts.

1. **Tutorials**  help you to get started with **respy**. They cover the basics and are
   designed for everyone new to the package and structural models. Although, the focus
   is on the implementation, you will find several cross-references to the theoretical
   concepts behind the models.

2. **Explanations** give detailed information to key topics and concepts which underlie
   the package.

3. **How-to Guides** are designed to provide detailed instructions for very specific
   tasks.

4. **Reference Guides** explain how **respy** is implemented. If you want to contribute
   to **respy** or if you are simply interested in the inner workings, you will find
   this section helpful. They assume that you are already familiar with **respy**.

.. toctree::
    :caption: Table of Contents
    :maxdepth: 1

    tutorials/index
    explanations/index
    how_to_guides/index
    reference_guides/index
    about_us
    projects
    api
    development/index
    changes
    replications/index
