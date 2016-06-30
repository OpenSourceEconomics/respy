Installation
============

You can install the baseline version of the **respy** package using the `Python Package Index <https://pypi.python.org/pypi>`_ (PyPI). 

.. code-block:: bash

   $ pip install respy

Please make sure you have a recent version of **pip** installed as well to ensure the proper install order of the dependencies. We draw on the standard *SciPy Stack*. In particular, we use the **NumPy**, **SciPy** and **pandas** library. In addition, we use **statsmodels**. The **respy** package is supported and tested for *Python 2.7* and *Python 3.x* on Linux systems.

We provide a pure *Python* implementation as our baseline. However, to address performance constraints, we also provide a scalar and parallel *Fortran* implementation. Provided some additional requirements are met, they will be automatically installed as well.

Fortran
-------

Please make sure that the *gfortran* compiler is available on your path and has access to the `Linear Algebra PACKage (LAPACK) <http://www.netlib.org/lapack/>`_. If so, just call a slightly modified version of the installation command.

.. code-block:: bash

   $ pip install --no-binary respy respy

The *--no-binary* flag is required for now to ensure a compilation of the *Fortran* source code during the build. 

Parallelism
-----------

We also support parallel computing using the `Message Passing Interface (MPI) <http://www.mpi-forum.org/>`_ library. This requires that you have a recent version of the `MPICH <https://www.mpich.org/>`_ implementation available on your compiler's search path that was build with shared/dynamic libraries.

Testing
-------

We encourage you to test a successful installation by running our test suite. If you have  **pytest** installed, go ahead and run the test suite to ensure a proper installation.

.. code-block:: bash

   $ python -c "import respy; respy.test()"

Feel free to pass any string for the usual **pytest** options into the test function. Depending on your installation, some tests might be skipped and some might actually fail as we compare the results from a pure *Python* implementation against the *Fortran* executables. We keep both implementations aligned as much as possible, but small numerical discrepancies might still arise. So, please just check the test output for any large discrepancies. If you find any, please let us know. 

