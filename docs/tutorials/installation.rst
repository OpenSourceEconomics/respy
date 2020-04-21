Installation
============

The **respy** package can be conveniently installed from the `Anaconda.org
<https://anaconda.org/>`_ or directly from its source files. We currently support only
Python 3.6+. We develop the package mainly on Linux and Windows, but the test battery
ensures compatibility with MacOS.


Anaconda.org
------------

The recommended way to install **respy** is via `conda <https://conda.io/>`_, the
standard package manager for scientific Python libraries. If conda is not installed on
your machine, please follow the `installation instructions
<https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_ of its user
guide.

With conda available on your path, installing **respy** is as simple as typing

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


Source Files
------------

You can download the sources directly from our `GitHub repository
<https://github.com/OpenSourceEconomics/respy>`_.

.. code-block:: bash

   $ git clone https://github.com/OpenSourceEconomics/respy.git

Once you obtained a copy of the source files, installing the package in editable mode is
straightforward.

.. code-block:: bash

   $ conda develop .


Test Suite
----------

Please make sure that the package is working properly by running our test suite using
``pytest``.

.. code-block:: python

    import respy as rp

    respy.test()
