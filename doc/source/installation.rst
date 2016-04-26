Installation
============

.. warning::

    The **respy** package is currently only working with *Python 3*. 

The **respy** package builds on the standard *SciPy Stack*. In particular, we use the **NumPy**, **SciPy** and **pandas** library. In addition, we require **statsmodels**. Please make sure that all these packages are properly installed. For more details on the package dependencies, please check out the requirements file on `GitHub <https://github.com/restudToolbox/package/blob/master/requirements.txt>`_

Please make sure you have a recent version of **pip** installed as well. Once these requirements are taken care of, installation is straightforward:

.. code-block:: bash

   $ pip install respy

If you have  **pytest** installed as well, then you can also run the test suite to ensure proper installation:

.. code-block:: bash

   $ python -c "import respy; respy.test()"

Some tests might actually fail as we test the results from a pure *Python* implementation against a *Fortran* executable. We keep both implementations aligned as much as possible, but small numerical discrepancies might still arise. So, just check the test output for any large discrepancies. If you find any of those, please let us know. More details about our testing efforts is available in our section on :ref:`implementation`

.. todo::

   There is also a dependency on LAPACK library. This needs to be handled more flexibly in the build process first. It is currently hard-coded in the *wscript*.

   We want the package to work with Python 2 and Python 3.