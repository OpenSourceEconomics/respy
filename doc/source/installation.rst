Installation
============

The **respy** package builds on the standard *SciPy Stack*. In particular, we use the **NumPy**, **SciPy** and **pandas** library. In addition, we use **statsmodels**. Please make sure that all those are properly installed. For more details on the dependencies, please check out the requirements file on `GitHub <https://github.com/restudToolbox/package/blob/master/requirements.txt>`_

If that is the case, the installation **respy** is straightforward.

.. code-block:: bash

   $ pip install respy

If you have  **pytest** installed, then you can also run the test suite to ensure proper installation:

.. code-block:: bash

   $ python -c "import respy; respy.test()"

.. todo::

   There is also a dependency on LAPACK library. This needs to be handled more flexibly in the build process first. It is currently hard-coded in the *wscript*.