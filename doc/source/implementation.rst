.. _implementation:

Implementation Details
======================

Numerical Methods
-----------------


Integration
"""""""""""

Integrals arise in the solution of the model and during the evaluation of the likelihood function. All are solved by Monte Carlo integration.

* For each agent in each time period, the **choice probabilities** are requires the approximated of a four-dimensional integral, see equation XXX. The integral is evaluated using the number of draws specified in the *SIMULATION* section of the initialization file.

* The calculation of the **expected future value** at each state requires the evaluation of a four dimensional integral, see equation XXX. The integral is evaluated using the number of draws specified in the *ESTIMATION* section of the initialization file.

Optimization
""""""""""""

The optimizer used for the estimation is set in the *ESTIMATION* section of the initialization file. Currently, the Powell and BFGS algorithms are available through their SCIPY implementations.

Approximation
"""""""""""""

The details for the **EMAX interpolation** are already discussed in detail :ref:`Eisenhauer (2016) <bibSection>`.

Program Design
--------------

We have learned a lot from the codes of the original codes. You can check them out `online <https://github.com/restudToolbox/package/tree/master/forensics>`_, where we also provide upgraded versions of the key modules.

.. todo::

   Provide an overview about the basic design of the program.

Test Battery
------------

You can have a look at our tests `online <https://github.com/restudToolbox/package/tree/master/respy/tests>`_. We also build a testing infrastructure, that allows to test the functionality of the **respy** package for numerous alternative parameterizations (`link <https://github.com/restudToolbox/package/tree/master/development/testing>`_). This setup, inspired by the idea of property based testing, has proven tremendously useful to check the validity of the code for numerous admissible parameterizations.



.. todo::

   Pitch the testing efforts.

