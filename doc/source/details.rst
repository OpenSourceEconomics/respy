Additional Details
==================


Output Files
------------

Depending on the user's request, the ``respy`` package creates several output files. Note that there is a difference between the estimation parameters in the files below and the model specification. The difference is in the parameters for the covariance matrix. During the estimation we iterate on a flattened version of the upper-triangular Cholesky decomposition. This ensures that the requirements for a valid covariance matrix, e.g. positive semidefiniteness and strictly positive variances, are always met as the optimizer tests the whole real line.

Simulation
""""""""""

.. _data.respy.dat:
* **data.respy.dat**

This file contains the agent choices and state experiences. The simulated dataset has the following structure.

    ======      ========================
    Column      Information
    ======      ========================
    1           agent identifier
    2           time period
    3           choice (1 = occupation A, 2 = occupation B, 3 = education, 4 = home)
    4           earnings (missing value if not working)
    5           work experience in occupation A
    6           work experience in occupation B
    7           years of schooling
    8           lagged schooling
    ======      ========================

* **data.respy.info**

This file provides descriptive statistics such as the choice probabilities and the wage distributions. It also prints out the underlying parameterization of the model.

* **sim.respy.log**

This file allows to monitor the progress of the simulation. It provides information about the seed used to sample the random components of the agents' state experience and the total number of simulated agents.

* **sol.respy.log**

This file allow to monitor the progress of the backward induction procedure. If the interpolation method is used during the backward induction procedure, the coefficient estimates and goodness of fit statistics are provided.

Estimation
""""""""""

* **est.respy.info**

This file allows to monitor the estimation as it progresses. It provides information about starting values, step values, and current values as well as the corresponding value of the criterion function.

* **est.respy.log**

This file documents details about the each of the evaluations of the criterion function. Most importantly, once estimation is completed, it provides the return message from the optimizer.

API Reference
-------------

The API reference provides detailed descriptions of ``respy`` classes and
functions. It should be helpful if you plan to extend ``respy`` with custom components.

.. class:: respy.RespyCls(fname)

    Class to process and manage the user's initialization file.

    :param str fname: Path to initialization file
    :return: Instance of RespyCls

    .. py:classmethod:: update_model_paras(x)

        Function to update model parameterization.

        :param numpy.ndarray x: Model parameterization

.. function:: respy.simulate(respy_obj)

    Simulate dataset of synthetic agents following the model specified in the
    initialization file.

    :param obj respy_obj: Instance of RespyCls class
    :return: Instance of RespyCls

.. function:: respy.estimate(respy_obj)

    Estimate a model based on a provided dataset and the model specified in the initialization file.

    :param obj respy_obj: Instance of RespyCls class

    :return: Model parameterization at final step
    :rtype: numpy.ndarray

    :return: Value of criterion function at final step
    :rtype: float
