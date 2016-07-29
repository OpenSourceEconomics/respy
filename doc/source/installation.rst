Installation
============

You can install the baseline version of the **respy** package using the `Python Package Index <https://pypi.python.org/pypi>`_ (PyPI).

.. code-block:: bash

   $ pip install respy

Please make sure you have a recent version of **pip** installed as well to ensure the proper install order of the dependencies. We draw on the standard *SciPy Stack*. In particular, we use the **NumPy**, **SciPy** and **pandas** library. In addition, we use **statsmodels**. The **respy** package is supported and tested for *Python 2.7* and *Python 3.x* on Linux systems.

Please make sure to test your installation. If you have  **pytest** installed, just go ahead and run the test suite:

.. code-block:: bash

   $ python -c "import respy; respy.test()"

We provide a pure *Python* implementation as our baseline. However, to address performance constraints, we also provide scalar and parallel *Fortran* implementations. If additional requirements are met, both are installed automatically.

... adding Fortran
------------------

Please make sure that the *gfortran* compiler is available on your path and it knows where to find the `Linear Algebra PACKage (LAPACK) <http://www.netlib.org/lapack/>`_. If so, just call a slightly modified version of the installation command.

.. code-block:: bash

   $ pip install --no-binary respy respy

The *--no-binary* flag is required for now to ensure a compilation of the *Fortran* source code during the build.

... adding Parallelism
----------------------

We also support parallel computing using the `Message Passing Interface (MPI) <http://www.mpi-forum.org/>`_ library. This requires that you have a recent version of its `MPICH <https://www.mpich.org/>`_ implementation available on your compiler's search path which was build with shared/dynamic libraries.
