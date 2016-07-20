.. _implementation:

Numerical Methods
-----------------

The simulation and estimation of the prototypical discrete choice dynamic programming model requires the use of several numerical components. We discuss each in turn.

Differentiation
"""""""""""""""

The scaling procedure and the derivative-based optimization algorithms require the approximation of the derivatives of the criterion function. We use a one-sided finite-difference approximation. The step-size can be controlled in the *DERIVATIVES* section of the initialization file.

Integration
"""""""""""

Integrals arise in the solution of the model and during the evaluation of the likelihood function. All are approximated by Monte Carlo integration. The same random draws are used for each integral.

* For each agent in each time period, the evaluation of the **choice probabilities** requires the approximation of a four-dimensional integral. The integral is evaluated using the number of draws specified in the *SIMULATION* section of the initialization file.

* The calculation of the **EMAX** at each state requires the evaluation of a four dimensional integral. The integral is evaluated using the number of draws specified in the *ESTIMATION* section of the initialization file.

Optimization
""""""""""""

The model is estimated using simulated maximum likelihood estimation (Albright, 1977)). The available optimizers depend on the version of the program. If you use the *Python* implementation, then the Powell and BFGS algorithms are available through their **SciPy** implementations. See the `SciPy  Documentation <http://docs.scipy.org>`_ for details. For the *FORTRAN*  implementation, we provide the `BFGS <https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm>`_ and `NEWUOA <https://en.wikipedia.org/wiki/NEWUOA>`_ (Powell, 2004) algorithms.

Function Approximation
""""""""""""""""""""""

The details for the **EMAX interpolation** are discussed in :ref:`Eisenhauer (2016) <bibSection>`.

Miscellaneous
"""""""""""""

We are using the `LAPACK <http://www.netlib.org/lapack>`_  for all numerical linear algebra. The pseudo-random numbers are generated using the *Mersenne Twister*.
