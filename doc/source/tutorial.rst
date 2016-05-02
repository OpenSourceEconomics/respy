Tutorial 
========

Let us explore the basic capabilities of the **respy** package in this tutorial (`script <https://github.com/restudToolbox/package/blob/master/example/example.py>`_).  We usually either want to simulate a synthetic sample form the  model or start an estimation run. Whatever the case, we always initialize an instance of the *RespyCls* first by passing in the path to the initialization file.
::

    from respy import RespyCls

    respy_obj = RespyCls('data_one.ini')

Now we can simulate a sample from the model::

    from respy import simulate

    simulate(respy_obj)

The sample is simulated with the parameters specified in the initialization file. During the simulation, several files will appear in the current working
directory.

* **logging.respy.sol.log**, logging information from the solution algorithm
* **logging.respy.sim.log**, logging information from the simulation algorithm

The names of the following files depend on that is specified for the filename in
the SIMULATION section of the model initialization file.

* **data.respy.dat**, simulated dataset with the agent choices and state experiences (:ref:`details <data.respy.dat>`)

* **data.respy.info**, basic descriptives of simulated dataset

* **data.respy.paras**, parameterization of model for simulated dataset (:ref:`details <data.respy.paras>`)

Now that we have some observed data, we can start an estimation. Here we are using the simulated data for the estimation. However, you can of also use other data sources. Just make sure they follow the layout of the simulated sample. The coefficient values in the initialization file serve as the starting values::

    from respy import estimate

    x, crit_val = estimate(respy_obj)

This directly returns the value of the coefficients at the final step of the optimizer as well as
the value of the criterion function. However, some additional files appear in the meantime.

* **optimization.respy.log**, logging information from optimizer

* **optimization.respy.info**, logging information for monitoring of estimation run  

The last file is continuously updated and provides information about the current parameterization, the starting values, and the the value at each step. Finally, the information about the model parameterization during optimization is continuously updated and written to disk (:ref:`details <paras.respy.log>`).

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