Quickstart
==========

Installation
------------

The ``respy`` package can be conveniently installed from the `Python Package Index
<https://pypi.org/>`_ (PyPI) or directly from its source files. We currently support
only Python 3.6+. We develop the package mainly on Linux systems, but the test battery
ensures compatibility with Windows and MacOS.

Python Package Index
^^^^^^^^^^^^^^^^^^^^

You can install the stable version of the package the usual way.

.. code-block:: bash

    $ pip install respy

We provide a pure Python implementation as our baseline. However, to address performance
constraints, we also maintain scalar and parallel Fortran implementations. If additional
requirements are met, both are installed automatically.

Source Files
^^^^^^^^^^^^

You can download the sources directly from our `GitHub repository
<https://github.com/OpenSourceEconomics/respy>`_.

.. code-block:: bash

   $ git clone https://github.com/OpenSourceEconomics/respy.git

Once you obtained a copy of the source files, installing the package in editable mode is
straightforward.

.. code-block:: bash

   $ pip install -e .

Test Suite
^^^^^^^^^^

Please make sure that the package is working properly by running our test suite using
``pytest``.

.. code-block:: bash

  $ python -c "import respy; respy.test()"
