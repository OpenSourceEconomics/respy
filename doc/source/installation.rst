Installation
============

.. warning::

    While we tested the installation on several UNIX machines, the process is not as robust to changes in the environment as we aim for.

The **respy** package is maintained for *Python 2.7* and *Python 3.x*. We build on the standard *SciPy Stack*. In particular, we use the **NumPy**, **SciPy** and **pandas** library. In addition, we require **statsmodels**. Please make sure that all these packages are properly installed. We are also using the *LAPACK* library, which needs to be available in your compiler's search path. For more details on the **respy** dependencies, please check out the requirements file on `GitHub <https://github.com/restudToolbox/package/blob/master/requirements.txt>`_

Please make sure you have a recent version of **pip** installed as well. Once these requirements are taken care of, installation is straightforward:

.. code-block:: bash

   $ pip install respy

If you have  **pytest** installed, go ahead and run the test suite to ensure a proper installation.

.. code-block:: bash

   $ python -c "import respy; respy.test()"

Feel free to pass any string for the usual **pytest** options into the test function. Some tests might actually fail as we test the results from a pure *Python* implementation against a *Fortran* executable. We keep both implementations aligned as much as possible, but small numerical discrepancies might still arise. So, please just check the test output for any large discrepancies. If you find any, please let us know. 

.. todo::

   More robust build process.