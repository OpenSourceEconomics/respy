Installation
============

The ``respy`` package can be conveniently installed from the `Python Package Index <https://pypi.python.org/pypi>`_ (PyPI) or directly from its source files. We currently support Python 2.7 and Python 3.3+. We develop the package on Linux systems, but it can also be installed on MacOS and Windows.

Python Package Index
--------------------

You can install the stable version of the package the usual way.

.. code-block:: bash

   $ pip install respy

We provide a pure Python implementation as our baseline. However, to address performance constraints, we also maintain scalar and parallel Fortran implementations. If additional requirements are met, both are installed automatically.

... adding Fortran
^^^^^^^^^^^^^^^^^^

Please make sure that the ``gfortran`` compiler is available on your path and it knows where to find the `Linear Algebra PACKage (LAPACK) <http://www.netlib.org/lapack/>`_ library.

On Ubuntu systems, both can be achieved by the following commands:

.. code-block:: bash

    $ sudo apt-get install gfortran
    $ sudo apt-get install libblas-dev liblapack-dev

If so, just call a slightly modified version of the installation command.

.. code-block:: bash

   $ pip install --no-binary respy respy

The *--no-binary* flag is required for now to avoid the use of Python Wheels and ensure a compilation of the Fortran source code during the build.

... adding Parallelism
^^^^^^^^^^^^^^^^^^^^^^

We use the `Message Passing Interface (MPI) <http://www.mpi-forum.org/>`_ library. This requires a recent version of its `MPICH <https://www.mpich.org/>`_ implementation available on your compiler's search path which was build with shared/dynamic libraries.

Source Files
------------

You can download the sources directly from our `GitHub repository <https://github.com/restudToolbox/package>`_.

.. code-block:: bash

   $ git clone https://github.com/restudToolbox/package.git

Once you obtained a copy of the source files, installing the package in editable model is straightforward.

.. code-block:: bash

   $ pip install -e .

Test Suite
----------

Please make sure that the package is working properly by running our test suite using ``pytest``.

.. code-block:: bash

  $ python -c "import respy; respy.test()"
