.. _implementation:

Numerical Details
-----------------

Integration
"""""""""""

Integrals arise in the solution of the model and during the evaluation of the likelihood function. All are solved by Monte Carlo integration. The same random draws are used for each integral.

* For each agent in each time period, the evaluation of the **choice probabilities** requires the approximation of a four-dimensional integral. The integral is evaluated using the number of draws specified in the *SIMULATION* section of the initialization file.

* The calculation of the **EMAX** at each state requires the evaluation of a four dimensional integral. The integral is evaluated using the number of draws specified in the *ESTIMATION* section of the initialization file.

Optimization
""""""""""""

The model is estimated using maximum likelihood estimation. The optimizer used for the optimization of the likelihood function is set in the *ESTIMATION* section of the initialization file. Currently, the Powell and BFGS algorithms are available through their **SciPy** implementations. If required, derivatives are approximated using the step size specified with the *eps* flag.

Approximation
"""""""""""""

The details for the **EMAX interpolation** are discussed in :ref:`Eisenhauer (2016) <bibSection>`.

Miscellaneous
"""""""""""""

We are using the *LAPACK* library for all numerical linear algebra. The pseudorandom numbers are generated using the *Mersenne Twister*.