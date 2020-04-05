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

``respy`` is an open source framework written in Python for the simulation and
estimation of some finite-horizon discrete choice dynamic programming models. The group
of models which can be currently represented in ``respy`` are called
Eckstein-Keane-Wolpin models (Aguirregabiria and Mira (2010))

What makes ``respy`` powerful is that it allows to build and solve structural models in
minutes whose development previously took years. The design of ``respy`` allows the
researcher to flexibly add the following components to her model.

- **Any number of two or more choices** (e.g., working alternatives, schooling, home
  production, retirement) where each choice may yield a wage, may allow for experience
  accumulation and can be constrained by time, a maximum amount of accumulated
  experience or other characteristics.

- Condition the decision of individuals on its **previous choices** or their labor
  market history.

- Adding a **finite mixture** with any number of subgroups to account for unobserved
  heterogeneity among individuals as developed by Keane and Wolpin (1997).

- **Any number of additional characteristics** (e.g., ability measures (Bhuller et al.
  (2020)), race (Keane and Wolpin (2000)), demographic variables) found in the data.

- Estimate **measurement errors** in wages using a Kalman Filter.

You can install ``respy`` via conda with

.. code-block:: bash

    $ conda config --add channels conda-forge
    $ conda install -c opensourceeconomics respy

Please visit our `online documentation <https://respy.readthedocs.io/en/latest/>`_ for
tutorials and other information.


.. Keep following section in sync with ./docs/additional_information/credits.rst.

Citation
--------

``respy`` was completely rewritten in the second release and evolved into a general
framework for the estimation of Eckstein-Keane-Wolpin models. Please cite it with

.. code-block::

    @Unpublished{Raabe2020,
      Title  = {respy - A Framework for the Simulation and Estimation of
                Eckstein-Keane-Wolpin Models.},
      Author = {Tobias Raabe},
      Year   = {2020},
      Url    = {https://github.com/OpenSourceEconomics/respy},
    }

Before that, ``respy`` was developed by Philipp Eisenhauer and provided a package for
the simulation and estimation of a prototypical finite-horizon discrete choice dynamic
programming model. At the heart of this release is a Fortran implementation with Python
bindings which uses MPI and OMP to scale up to HPC clusters. It is accompanied by a pure
Python implementation as teaching material. If you use ``respy`` up to version 1.2.1,
please cite it with

.. code-block::

    @Software{Eisenhauer2019,
      Title  = {respy - A Package for the Simulation and Estimation of a prototypical
                finite-horizon Discrete Choice Dynamic Programming Model.},
      Author = {Philipp Eisenhauer},
      Year   = {2019},
      DOI    = {10.5281/zenodo.3011343},
      Url    = {https://doi.org/10.5281/zenodo.3011343}
    }

We appreciate citations for ``respy`` because it helps us to find out how people have
been using the package and it motivates further work.


References
----------

Aguirregabiria, V., & Mira, P. (2010). `Dynamic discrete choice structural models: A
survey <https://doi.org/10.1016/j.jeconom.2009.09.007>`_. Journal of Econometrics,
156(1), 38-67

Bhuller, M., Eisenhauer, P. and Mendel, M. (2020). The Option Value of Education.
*Working Paper*.

Keane, M. P. and  Wolpin, K. I. (1994). `The Solution and Estimation of Discrete Choice
Dynamic Programming Models by Simulation and Interpolation: Monte Carlo Evidence
<https://doi.org/10.2307/2109768>`_. *The Review of Economics and Statistics*, 76(4):
648-672.

Keane, M. P. and Wolpin, K. I. (1997). `The Career Decisions of Young Men
<https://doi.org/10.1086/262080>`_. *Journal of Political Economy*, 105(3): 473-522.

Keane, M. P., & Wolpin, K. I. (2000). `Eliminating race differences in school attainment
and labor market success <https://www.journals.uchicago.edu/doi/abs/10.1086/209971>`_.
Journal of Labor Economics, 18(4), 614-652.
