Basic Usage
===========

We now illustrate the basic capabilities of the ``respy`` package in a simple tutorial.

.. todo:: The following link refers to the test/resources dir since there is no longer any special directory for the tutorial files. Would it be better to separate this files in a specific example directory? Additionally there is no longer an example.ini initalization file. The link under prerequisites still refers to the master branch, sinse there is no example.ini file in janosg.

All the material is available `online <https://github.com/OpenSourceEconomics/respy/tree/janosg/respy/tests/resources>`__.

Prerequisites
-------------

**Python**

To complete the current tutorial a Pyhon v. 3.4 or newer installation is required.

**Initialization File**

In order to perform simulation and/or estimation using ``respy`` package an initialization file containing initial parameter values, important data set dimentions, arguments for the optimization algorythms, etc. needs to be specified. Details on the components of the initialization file are presented in the section :ref:`Model Specification <specification>`.
Please note that the initialization file should be in your current working directory or another accessible directory when executing the commands and scripts discussed below.

Example
-------
Now we can explore the basic functionalities of the ``respy`` package based on a simple example.

**Simulation and Estimation**

Execution of the short script below in a python interpreter performs the following:
* calls for the usage of the ``respy`` package,
* defines a model in ``respy``  based on the initialization file,
* simulates a data set of agents' choices and state experienes for the specified number of agents and periods based on the parameters values in the initialization file,
* uses the simulated data as an imput data set to estimate requested parameters based on starting values specialized in the initialization file,
* simulates a data set for the specified number of agents and periods based on the parameters values in the initialization file.


::
    """ This module simulates and estimates a model based on an initial specification
    """

    # Import the package
    import respy

    # Create example.ini and save directory as working directory
    import os
    os.chdir('your_dir')

    # Initialize an instance of the RespyCls 
    import respy
    respy_obj = respy.RespyCls('example.ini')

    # Simulate a sample from the specified model
    respy.simulate(respy_obj)

    # Estimate the model using the simulated data as an observed sample
    x, crit_val = respy.fit(respy_obj)

    # Simulate a semple based on estimated parameters
    respy_obj.update_model_paras(x)
    respy.simulate(respy_obj)

The simulation and estimation functionalities of the ``respy`` package can also be used seperately. To perform a simulation only an initialization file, as discussed above, is required. To directly estimate the model parameters your working directory has to contain the initialization file and your data set. Here we are using the simulated data for the estimation. However, you can of course also use other data sources. Just make sure they follow the layout of the simulated sample as visible in ``data.respy.dat``. For more information on the required structure of the dataset see :ref:`Model Specification <specification>`. The coefficient values in the initialization file serve as the starting values.


**Output Files**

During the script execution, several files will appear in the current working directory.
First, we outline the files generated during the initial simulation.

* **data.respy.sol**

Records the progress of the backward induction procedure. If the interpolation method is used during the backward induction procedure, the coefficient estimates and goodness of fit statistics are provided.

* **data.respy.pkl**

This file is an instance of the ``RespyCls`` and contains detailed information about the solution of model such as the :math:`E\max` of each state for example. For details, please consult the `source code <https://github.com/restudToolbox/package/blob/master/respy/clsRespy.py>`_ directly. It is created if persistent storage of results is requested in the *SOLUTION* section of the initialization file.

* **data.respy.sim**

Allows to monitor the progress of the simulation. It provides information about the seed used to sample the random components of the agents' state experience and the total number of simulated agents.

* **data.respy.dat**

Contains the simulated data on agents' choices and state experiences. It has the following structure:

    =======     ========================
    Column      Information
    =======     ========================
    1           agent identifier
    2           time period
    3           choice (1 = Occupation A, 2 = Occupation B, 3 = education, 4 = home)
    4           wages (missing value if not working)
    5           work experience in Occupation A
    6           work experience in Occupation B
    7           years of schooling
    8           lagged choice
    9           type number (0 for the whole column, if homogenous agents)
    10 - 13     total rewads - all components
    14 - 17     systematic reward - no shock
    18 - 21     shock reward - shock component
    22          discount rate
    23 - 24     general reward - non-monetary rewards and non- common rewards, example cm1 cm2 and alpha for occupation A
    25          common reward - indicators assoc with beta 1 and beta 2
    26 - 29     immediate reward - period reward
    =======     ========================

* **data.respy.info**

Provides descriptive statistics such as the choice probabilities, the transition matrix, number of agents per period and occupation, and the rspctive wage distributions. It also prints out the underlying parameterization of the model.

Second, we turn to the estimation output. The fit procedure directly returns the value of the coefficients at the final step of the optimizer, as well as the value of the criterion function. However, some additional files appear in the meantime. 

* **est.respy.info**

Allows to monitor the estimation as it progresses. It provides information about starting values, step values, and current values as well as the corresponding value of the criterion function.

* **est.respy.log**

Documents details about the estimation procedure. Provides informaton on the precondigitoning of the parameters including the original parameter value, the scailing factor and the rescaled parameter. Further, details about each of the evaluations of the criterion function are included. Most importantly, once an estimation is completed, it provides the return message from the optimizer.

Third, additional information is provided in two further generated files:

* **scaling.respy.out**

* **solution.respy.pkl**

Finally, when a second simulation is performed, now based on the parameter estimates, the existing simulation output files are replaced by new ones referring to the current simulation run.

