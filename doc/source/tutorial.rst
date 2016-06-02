Tutorial 
========


Model Specification
-------------------

The model is specified in an initialization file. For an example, check out the first specification analyzed in the original paper (Table 1) `online <https://github.com/restudToolbox/package/blob/master/example/data_one.ini>`_.

We turn to each of the ingredients in more details.

**BASICS**

=======     ======      ================== 
Key         Value       Interpretation      
=======     ======      ==================  
periods      int        number of periods  
delta        float      discount factor
=======     ======      ================== 

We now turn to the specification of the two alternative occupations. There are two small differences between the setup in the original paper and the parameterization of the occupations. First, all coefficients enter the return function with a positive sign, while the squared terms enter with a minus in the original paper. Second, the order of covariates is fixed across the two occupations. In the original paper, own experience always comes before other experience. These comments are valid for occupation A and occupation B.

**OCCUPATION A**

=======     ======    ================== 
Key         Value       Interpretation      
=======     ======    ================== 
coeff       float       intercept  
coeff       float       return to schooling
coeff       float       experience occupation A, linear
coeff       float       experience occupation A, squared
coeff       float       experience occupation B, linear  
coeff       float       experience occupation B, squared
=======     ======    ================== 

**OCCUPATION B**

=======     ======    ================== 
Key         Value       Interpretation      
=======     ======    ================== 
coeff       float       intercept  
coeff       float       return to schooling
coeff       float       experience occupation A, linear
coeff       float       experience occupation A, squared
coeff       float       experience occupation B, linear  
coeff       float       experience occupation B, squared
=======     ======    ================== 

**EDUCATION**

======= ======    ==========================
Key     Value       Interpretation      
======= ======    ========================== 
coeff    float      consumption value
coeff    float      tuition cost
coeff    float      adjustment cost
max      int        maximum level of schooling
start    int        initial level of schooling
======= ======    ========================== 

Again, there is a small difference between the setup in the original paper. There is no automatic change in sign for the tuition and adjustment costs. Thus, a \$1,000 tuition cost must be specified as -1000.

**HOME**

======= ======      ==========================
Key     Value       Interpretation      
======= ======      ========================== 
coeff    float      mean value of non-market alternative
======= ======      ========================== 

**SHOCKS**

======= ======      ==========================
Key     Value       Interpretation      
======= ======      ========================== 
coeff    float      :math:`\sigma_{1}`
coeff    float      :math:`\sigma_{12}`
coeff    float      :math:`\sigma_{13}`
coeff    float      :math:`\sigma_{14}`
coeff    float      :math:`\sigma_{2}`
coeff    float      :math:`\sigma_{23}`
coeff    float      :math:`\sigma_{24}`
coeff    float      :math:`\sigma_{3}`
coeff    float      :math:`\sigma_{34}`
coeff    float      :math:`\sigma_{4}`
======= ======      ========================== 

**SOLUTION**

=======     ======      ==========================
Key         Value       Interpretation      
=======     ======      ========================== 
draws       int         number of draws for EMAX approximation
store       bool        store results
seed        int         random seed for the EMAX approximation
=======     ======      ========================== 

**SIMULATION**

=======     ======      ==========================
Key         Value       Interpretation      
=======     ======      ==========================
file        str         file to print simulated sample, *.dat* added automatically
agents      int         number of simulated agents
seed        int         random seed for agent experience
=======     ======      ==========================

**ESTIMATION**

==========      ======      ==========================
Key             Value       Interpretation      
==========      ======      ==========================
file            str         file to read observed sample, *.dat* added automatically
tau             float       smoothing window
agents          int         number of agents to read from sample
maxiter         int         maximum number of iterations for optimizer
seed            int         random seed for choice probability approximation
optimizer       str         optimizer to use
==========      ======      ==========================

**PROGRAM**

=======     ======      ==========================
Key         Value       Interpretation      
=======     ======      ==========================
debug       bool        flag to use debug mode
version     str         program version
=======     ======      ==========================

**INTERPOLATION**

=======     ======      ==========================
Key         Value       Interpretation      
=======     ======      ==========================
points      int         number of interpolation points
apply       bool        flag to use interpolation
=======     ======      ==========================

Two alternative optimization algorithms are available for the estimation. In both cases, we use the **SciPy** package. The implementation details are available `here <http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.minimize.html>`_

**SCIPY-BFGS**

=======     ======      ==========================
Key         Value       Interpretation      
=======     ======      ==========================
gtol        float       gradient norm must be less than gtol before successful termination 
epsilon     float       step size for numerical approximation of first derivatives 
=======     ======      ==========================

**SCIPY-POWELL**

=======     ======      ==========================
Key         Value       Interpretation      
=======     ======      ==========================
maxfun      int         maximum number of function evaluations to make
ftol        float       relative error in func(xopt) acceptable for convergence      
xtol        float       line-search error tolerance         
=======     ======      ==========================


Simulation and Estimation
-------------------------

Let us explore the basic capabilities of the **respy** package in this tutorial (`script <https://github.com/restudToolbox/package/blob/master/example/example.py>`_).  We usually either want to simulate a synthetic sample from the  model or start an estimation run. Whatever the case, we always initialize an instance of the *RespyCls* first by passing in the path to the initialization file.
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

The names of the following files depend on the specified filename in the SIMULATION section of the model initialization file, where we specified *data.respy*. The extensions are automatically added.

* **data.respy.dat**, simulated dataset with the agent choices and state experiences (:ref:`details <data.respy.dat>`)

* **data.respy.info**, basic descriptives of simulated dataset

* **data.respy.paras**, parameterization of model for simulated dataset (:ref:`details <data.respy.paras>`)

Now that we have some observed data, we can start an estimation. Here we are using the simulated data for the estimation. However, you can of course also use other data sources. Just make sure they follow the layout of the simulated sample and remember that the *.dat* extension will be added automatically to the filename specified in the ESTIMATION section. The coefficient values in the initialization file serve as the starting values::

    from respy import estimate

    x, crit_val = estimate(respy_obj)

This directly returns the value of the coefficients at the final step of the optimizer as well as
the value of the criterion function. However, some additional files appear in the meantime.

* **optimization.respy.log**, logging information from optimizer

* **optimization.respy.info**, logging information for monitoring of estimation run  

The last file is continuously updated and provides information about the current parameterization, the starting values, and the value at each step. Finally, the information about the model parameterization during optimization is continuously updated and written to disk (:ref:`details <paras.respy.log>`).

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