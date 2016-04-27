Tutorial 
========

Let us explore the basic capabilities of the **respy** package in this tutorial (`script <https://github.com/restudToolbox/package/blob/master/example/example.py>`_).

We usually either want to simulate a synthetic sample form the  model or start an estimation run. At the heart of our interaction with the **respy** package is the *RespyCls*. This classes processes the user's model specification. We can initialize an instance of the class by passing in the location of the model initialization file.
::

    from respy import RespyCls

    respy_obj = RespyCls('data_one.ini')

Now we can simulate a sample from the model::

    from respy import simulate

    simulate(respy_obj)

The sample is simulated with the parameters specified in the initialization file. During the simulation, several files will appear in the current working
directory.

* **logging.respy.sol.log**

    This file contains logging information from the solution algorithm

* **logging.respy.sim.log**

    This file contains logging information from the simulation algorithm. It is updated as the simulation progresses and also reports the random seed for the random components of the agents' state experiences.

The names of the following files depend on that is specified for the filename in
the SIMULATION section of the model initialization file.

* **data.respy.dat**
    
    This file contains the agent choices and state experiences. The simulated dataset has the following structure.
    
    ======      ========================      
    Column      Information
    ======      ========================      
    1           agent identifier     
    2           time period     
    3           choice, 1 = work A, 2 = work B, 3 = education, 4 = home)     
    4           earnings (missing value if not working)     
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
    1 -  6      occupation A      
    7 - 12      occupation B     
    12 - 15     education     
    16          home     
    16 - 26     cholesky factors     
    =======     ========================

    Note, that the last ten coefficients do refer to the Cholesky factors of the
    covariance matrix of the shocks and not the covariance matrix itself. 

Now that we have some observed data, we can start an estimation. Here we are using the simulated data for the estimation. However, you can of also use other data sources. Just make sure they follow the layout of the simulated sample. The coefficient values in the initialization file serve as the starting values::

    from respy import estimate

    x, crit_val = estimate(respy_obj)

This directly returns the value of the coefficients at the final step of the optimizer as well as
the value of the criterion function. However, some additional files appear in the meantime.

* **optimization.respy.log**

The provides some information about each step of the optimizer and a final message from the optimizer about convergence. See the **SciPy** documentation for details on the convergence message.

* **optimization.respy.info**

This file is key to monitor the progress of the estimation run. It is continuously updated and provides information about the current parameterization, the starting values, and the the value at each step.

Finally, the parameters are written to disk:

* **paras_curre.respy.log**, current candidate parameters

* **paras_start.respy.log**, parameters at the start of the optimization

* **paras_steps.respy.log**, parameters at last step of the optimizer

We can now simulate a sample using the estimated parameters, but updating the instance of the *RespyCls* with the parameters returned from the estimation routine.
::
    respy_obj.update_model_paras(x)

    respy.simulate(respy_obj)


Recomputing Keane & Wolpin (1994)
---------------------------------

Just using the capabilities outlined so far, it is straightforward to compute some of the key results in the original paper with a simple script::

    #!/usr/bin/env python
    """ This module recomputes some of the key results of Keane & Wolpin (1994).
    """

    import respy

    # We can simply iterate over the different model specifications outlined in
    # Table 1 of their paper.
    for spec in ['data_one.ini', 'data_two.ini', 'data_three.ini']:

        # Process relevant model initialization file
        respy_obj = respy.RespyCls(spec)

        # Let us simulate the datasets discussed on the page 658.
        respy.simulate(respy_obj)

        # To start estimations for the Monte Carlo exercises. For now, we just
        # evaluate the model at the starting values, i.e. maxiter set to zero in
        # the initialization file.
        respy.estimate(respy_obj)

You can download the three initialization files `here <https://github.com/restudToolbox/package/tree/master/forensics/inits>`_. In an earlier working paper version of their paper (`online <https://www.minneapolisfed.org/research/staff-reports/the-solution-and-estimation-of-discrete-choice-dynamic-programming-models-by-simulation-and-interpolation-monte-carlo-evidence>`_), the original authors provide a full account of the choice distributions for all three specifications. The results from the recomputation line up well with their reports.