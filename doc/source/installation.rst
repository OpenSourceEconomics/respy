Installation
============

We provide instructions to install the **respy** package using the `Python Package Index <https://pypi.python.org/pypi>`_ and directly from source. We encourage you to test a successful installation by running our test suite. The **respy** package is supported and tested for *Python 2.7* and *Python 3.x* on Linux. We draw on the standard *SciPy Stack*. In particular, we use the **NumPy**, **SciPy** and **pandas** library. In addition, we use **statsmodels**. Please make sure that all these packages are properly installed. We are also using the *LAPACK* library, which needs to be available in your *gfortran* search path. 

As an option, we also support parallel computing using the `Message Passing Interface (MPI) <http://www.mpi-forum.org/>`_ library. During the configuration step, we will automatically determine if you have the `MPICH <https://www.mpich.org/>`_ implementation installed. If so, the parallel executables will be created during the build. 


Python Package Index
--------------------

Please make sure you have a recent version of **pip** installed as well to ensure the proper install order of the dependencies. Once these requirements are taken care of, installation is straightforward.

.. code-block:: bash

   $ pip install --no-binary respy respy

The *--no-binary* flag is required for now to ensure a compilation of the *FORTRAN* libraries. 

Sources
-------

For the most control during the installation process, please download the sources directlty from our `online repository <https://github.com/restudToolbox/package>`_. We use `waf <https://waf.io/>`_ as our build automation tool.


Testing
-------

If you have  **pytest** installed, go ahead and run the test suite to ensure a proper installation.

.. code-block:: bash

   $ python -c "import respy; respy.test()"

Feel free to pass any string for the usual **pytest** options into the test function. Some tests might actually fail as we compare the results from a pure *Python* implementation against a *Fortran* executable. We keep both implementations aligned as much as possible, but small numerical discrepancies might still arise. So, please just check the test output for any large discrepancies. If you find any, please let us know. 

