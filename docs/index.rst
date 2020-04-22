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
estimation of some finite-horizon discrete choice dynamic programming models.

With conda available on your path, installing **respy** is as simple as typing

.. code-block:: bash

    $ conda config --add channels conda-forge
    $ conda install -c opensourceeconomics respy

The documentation is structured in four parts.

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
    :maxdepth: 1

    tutorials/index
    explanations/index
    how_to_guides/index
    reference_guides/index


If you are looking for other resources, you might find them here.

.. toctree::
    :maxdepth: 1

    api
    about_us
    projects
    development/index
    replications/index
    changes
