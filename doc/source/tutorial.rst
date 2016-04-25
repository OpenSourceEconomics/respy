Tutorial 
========

Let us explore the basic capabilities of the **respy** package in this tutorial. The script is available `online <https://github.com/restudToolbox/package/blob/master/example/example.py>`_.


As a user of the **respy** package, we usually either want to simulate a synthetic sample form the 
model or start an estimation run. At the heart of our interaction with the **respy** package is 
the *RobupyCls*. This classes processes the user's model specification. We can initialize an instance of the class by passing in the location of the model initialization file.
::

    from respy import RespyCls

    respy_obj = RespyCls('data_one.ini')

This then allows us to simulate a sample from the model::

    from respy import simulate

    simulate(respy_obj)

The sample is simulated with the parameters specified in the initialization file, which are discussed in
Section :ref:`specification`.

During the simulation, several files will appear in the current working
directory.

* **logging.respy.sol.log**

    This file contains logging information from the solution algorithm

* **logging.respy.sim.log**

    This file contains logging information from the simulation algorithm. It is
    updated as the simulation progresses and also reports the random seed for
    the agent state experiences.

The names of the following files depend on that is specified for the filename in
the SIMULATION section of the model initialization file.

* **data.respy.dat**
    
    This file contains the agent choices and state experiences. The dataset has
    the following structure.
    
    ======      ========================      
    Column      Information
    ======      ========================      
    1           agent identifier     
    2           time period     
    3           Choice, 1 = Work A, 2 = Work B, 3 = Education, 4 = Home)     
    4           Earnings (missing value if not working)     
    5           work experience in occupation A     
    6           work experience in occupation B     
    7           years of schooling     
    8           lagged schooling     
    ======      ========================

* **data.respy.info**

    This file provides some basic descriptive information about the simulated
    agent population such as aggregate choice probabilities and wage
    distributions.

* **data.respy.paras**

    This file contains the coefficients used for the simulation of the agent
    population.

    =======     ========================      
    Lines       Coefficients
    =======     ========================       
    1 -  6      Occupation A      
    7 - 12      Occupation B     
    12 - 15     Education     
    16          Home     
    16 - 26     Cholesky Factors     
    =======     ========================

    Note, that the last ten coefficients do refer to the Cholesky factors of the
    covariance matrix of the shocks and not the covariance matrix directly. The
    Cholesky factors are in order of a flattened upper triangular matrix.

Now that we have some observed data, we can start an estimation. The coefficient values in the initialization file serve as the starting values::

    from respy import estimate

    x, crit_val = estimate(respy_obj)

This directly returns the value of the coefficients at the final step of the optimizer as well as
the value of the criterion function. However, some additional files appear in the meantime.


* **optimization.respy.log**

The provides some information about each step of the optimizer and a final message from the optimizer about convergence.

* **optimization.respy.info**

This file is key to monitor the progress of the estimation run. It is continuously updated and provides information about the current parameterization, the starting values, and the the value at each step.

Finally, the parameters are written to disk:

* *paras_curre.respy.log*, current candidate parameters

* *paras_start.respy.log*, parameters at the start of the optimization

* *paras_steps.respy.log*, parameters at last step of the optimizer