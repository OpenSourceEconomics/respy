Tutorial
========

Let us now illustrate the basic capabilities of the **respy** package. We start by setting up a model specification and then turn to some examples.

Model Specification
-------------------

The model is specified in an initialization file that is processed by the package. For an example, check out the first parameterization analyzed by Keane & Wolpin (1994) `online <https://github.com/restudToolbox/package/blob/master/respy/tests/resources/kw_data_one.ini>`_. Let us discuss each of the elements in more detail.

**BASICS**

=======     ======      ==================
Key         Value       Interpretation
=======     ======      ==================
periods      int        number of periods
delta        float      discount factor
=======     ======      ==================

We now turn to the specification of the reward functions for the two alternative occupations. There are two small differences compared to the original paper. First, all coefficients enter the return function with a positive sign, while the squared terms enter with a minus in the original paper. Second, the order of covariates is fixed across the two occupations. In the original paper, own experience always comes before other experience.

**OCCUPATION A**

=======     ======    ==============
Key         Value     Interpretation
=======     ======    ==============
coeff       float     intercept
coeff       float     return to schooling
coeff       float     experience occupation A, linear
coeff       float     experience occupation A, squared
coeff       float     experience occupation B, linear
coeff       float     experience occupation B, squared
=======     ======    ==============

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

If requested, the results from the solution are available in the file **solution.respy.pkl** for further inspection.

**SIMULATION**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
file        str         file to print simulated sample
agents      int         number of simulated agents
seed        int         random seed for agent experience
=======     ======      ==========================

**ESTIMATION**

==========      ======      ==========================
Key             Value       Interpretation
==========      ======      ==========================
file            str         file to read observed sample
tau             float       smoothing window for simulated probabilities
agents          int         number of agents to read from sample
draws           int         number of draws for approximation of choice probabilities
maxfun          int         maximum number of function evaluations
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

In general, two versions of the program are available. Either your can request the baseline *Python* implementation or *Fortran* for better performance.

**PARALLELISM**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
flag        bool        flag to use parallel executable
procs       int         number of processors
=======     ======      ==========================

**INTERPOLATION**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
points      int         number of interpolation points
flag        bool        flag to use interpolation
=======     ======      ==========================

**DERIVATIVES**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
version     str         approximation scheme
eps         float       step size
=======     ======      ==========================

Whenever numerical derivatives are required, then the details are specified here. Currently, only forward finite-differences are available.

**SCALING**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
flag        bool        apply scaling to criterion function
minimum     float       minimum value for gradient approximation
=======     ======      ==========================


The implemented optimization algorithms vary with the program's version. If you request the *PYTOHN* version of the program, you can choose from the **SciPy** implementations of the BFGS and POWELL algorithm. Their implementation details are available `here <http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.minimize.html>`_. For *FORTRAN*, we implemented the BFGS algorithm as well and the NEWUOA. For details about the algorithms, please see the section on implementation. Note that only the optimizer used for an estimation needs to be fully specified.

**SCIPY-BFGS**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
gtol        float       gradient norm must be less than gtol before successful termination
maxiter     int         maximum number of iterations
=======     ======      ==========================

**SCIPY-POWELL**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
maxfun      int         maximum number of function evaluations to make
ftol        float       relative error in func(xopt) acceptable for convergence
xtol        float       line-search error tolerance
=======     ======      ==========================

**FORT-BFGS**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
gtol        float       gradient norm must be less than gtol before successful termination
maxiter     int         maximum number of iterations
=======     ======      ==========================

**FORT-NEWUOA**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
maxfun      float       maximum number of function evaluations
npt         int         number of points for approximation model
rhobeg      float       starting value for size of trust region
rhoend      float       minimum value of size for trust region
=======     ======      ==========================

If you perform an estimation using the *NEWUOA* algorithm, the maximum number of function evaluations is determined by the minimum of the two flags specified in this section and the *ESTIMATION* block.

Examples
--------

Let us explore the basic capabilities of the **respy** package with a couple of examples. All the required material is available `online <https://github.com/restudToolbox/package/tree/master/example>`_.

**Simulation and Estimation**

We usually either want to simulate a synthetic sample from the model or start an estimation run. Whatever the case, we always initialize an instance of the *RespyCls* first by passing in the path to the initialization file.
::

    import respy

    respy_obj = respy.RespyCls('example.ini')

Now we can simulate a sample from the specified model::

    respy.simulate(respy_obj)

During the simulation, several files will appear in the current working directory. **sol.respy.log** allows to monitor the progress of the solution algorithm, while the actual simulation can be followed in  **sim.respy.log**. The names of the following files depend on the specified filename in the SIMULATION section of the model initialization file, where we specified *data.respy.dat*. The simulated dataset with the agent choices and state experiences is stored in **data.respy.dat** and **data.respy.info** provides some basic descriptives of the simulated dataset.

Now that we have some observed data, we can start an estimation. Here we are using the simulated data for the estimation. However, you can of course also use other data sources. Just make sure they follow the layout of the simulated sample. The coefficient values in the initialization file serve as the starting values::

    x, crit_val = respy.estimate(respy_obj)

This directly returns the value of the coefficients at the final step of the optimizer as well as the value of the criterion function. However, some additional files appear in the meantime. Monitoring the estimation is best done using **est.respy.info** and more details are in **est.respy.log**.

We can now simulate a sample using the estimated parameters, but updating the instance of the *RespyCls* with the parameters returned from the estimation routine.
::
    respy_obj.update_model_paras(x)

    respy.simulate(respy_obj)

**Recomputing Keane & Wolpin (1994)**

Just using the capabilities outlined so far, it is straightforward to compute some of the key results in the original paper with a simple script::

    #!/usr/bin/env python
    """ This module recomputes some of the key results of Keane & Wolpin (1994).
    """

    import respy

    # We can simply iterate over the different model specifications outlined in
    # Table 1 of their paper.
    for spec in ['kw_data_one.ini', 'kw_data_two.ini', 'kw_data_three.ini']:

        # Process relevant model initialization file
        respy_obj = respy.RespyCls(spec)

        # Let us simulate the datasets discussed on the page 658.
        respy.simulate(respy_obj)

        # To start estimations for the Monte Carlo exercises. For now, we just
        # evaluate the model at the starting values, i.e. maxfun set to zero in
        # the initialization file.
        respy_obj.unlock()
        respy_obj.set_attr('maxfun', 0)
        respy_obj.lock()

        respy.estimate(respy_obj)

You can download the all the material for the recomputation `here <https://github.com/restudToolbox/package/tree/master/respy/tests/resources>`_. In an earlier working paper version of their paper (`online <https://www.minneapolisfed.org/research/staff-reports/the-solution-and-estimation-of-discrete-choice-dynamic-programming-models-by-simulation-and-interpolation-monte-carlo-evidence>`_), the original authors provide a full account of the choice distributions for all three specifications. The results from the recomputation line up well with their reports.
