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


``respy``  is an open-source Python package for the simulation and estimation of a
prototypical finite-horizon discrete choice dynamic programming model. We build on the
baseline model presented in:

    Keane, M. P. and  Wolpin, K. I. (1994). `The Solution and Estimation of Discrete
    Choice Dynamic Programming Models by Simulation and Interpolation: Monte Carlo
    Evidence <https://doi.org/10.2307/2109768>`_. *The Review of Economics and
    Statistics*, 76(4): 648-672.

You can install ``respy`` via conda with

.. code-block:: bash

    $ conda config --add channels conda-forge
    $ conda install -c opensourceeconomics respy

Please visit our `online documentation <https://respy.readthedocs.io/en/latest/>`_ for
tutorials and other information.


Citation
--------

If you use respy for your research, do not forget to cite it with

.. code-block::

    @Unpublished{The respy Team,
      Title  = {respy - A Framework for the Estimation of some DCDP Models.},
      Author = {The respy Team},
      Year   = {2015},
      Url    = {https://github.com/OpenSourceEconomics/respy},
    }
