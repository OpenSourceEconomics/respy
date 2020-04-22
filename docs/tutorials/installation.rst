Installation
============

To use **respy** in the following tutorials, you need the following three components.

Anaconda
--------

The Anaconda distribution is a bundle of compatible Python packages. It also includes
``conda`` which is a package manager to install, update, and remove packages. You can
also manage environments with ``conda`` which are a collection of packages you need for
a project.

The installation instructions for multiple platforms can be found `here
<https://docs.anaconda.com/anaconda/install/>`_.


Jupyter Lab
-----------

Jupyter Lab is an IDE (integrated development environment) for literate programming
meaning that the notebook display code and text alongside each other in a pleasant way.

Jupyter Lab can be installed with

.. code-block:: bash

    $ conda install jupyterlab

Although `this tutorial <https://realpython.com/jupyter-notebook-introduction/>`_  is
dedicated to Jupyter notebooks, the same instructions apply to Jupyter Lab which will in
the long-run supersede Jupyter notebooks.


respy
-----

The recommended way to install **respy** is via `conda <https://conda.io/>`_, the
standard package manager for scientific Python libraries. With conda available on your
path, installing **respy** is as simple as typing

.. code-block:: bash

    $ conda config --add channels conda-forge
    $ conda install -c opensourceeconomics respy

in a command shell. The whole package repository can be found under
https://anaconda.org/OpenSourceEconomics/respy.

As **respy** relies heavily on ``pandas``, you might also want to install their
`recommended dependencies <https://pandas.pydata.org/pandas-docs/stable/getting_started/
install.html#recommended-dependencies>`_ to speed up internal calculations done with
`pd.eval <https://pandas.pydata.org/pandas-docs/stable/user_guide/enhancingperf.html
#expression-evaluation-via-eval>`_.

.. code-block:: bash

    conda install -c conda-forge bottleneck numexpr
