.. _model-specification:

Model specification
===================

In the following, we discuss the model specification in greater detail.

Overview
--------

The model is specified in an initialization file. In case the initialization file is
used to simulate a data set, the data generation is based on the chosen parameters. As
soon as the estimation procedure is invoked, the values specified in the initialization
file are used as starting values for the optimization. Now, let us discuss each of its
elements in more detail.

**BASICS**

=======     ======      ==================
Key         Value       Interpretation
=======     ======      ==================
periods      int        number of periods
delta        float      discount factor
=======     ======      ==================

**COMMON**

=======     ================    ======      ===============================
Key         Notation            Value       Interpretation
=======     ================    ======      ===============================
coeff       :math:`\beta_1`     float       return to hight school degree
coeff       :math:`\beta_2`     float       return to college education
=======     ================    ======      ===============================

.. Warning::

    There are two small differences compared to Keane and Wolpin (1997). First, all
    coefficients enter the return function with a positive sign, while the squared terms
    enter with a minus in the original paper. Second, the order of covariates is fixed
    across the two occupations. In the original paper, own experience always comes
    before other experience.

**OCCUPATION A**

.. include:: occ_a_params.rst

**OCCUPATION B**

.. include:: occ_b_params.rst

**OCCUPATION C**

.. warning::
    This is not implemented yet!

.. include:: occ_c_params.rst

**EDUCATION**

.. include:: occ_edu_params.rst

In addition to the parameters of the reward function, the following attributes of the
model have to be specified for education:

=======     ===================     ======    ==========================
Key         Notation                Value     Interpretation
=======     ===================     ======    ==========================
start       not present in KW       int       initial level of schooling
share       not present in KW       int       share of agents with respective
                                              initial level of schooling
lagged      :math:`d_4(15)`         int       was in education last year

start       not present in KW       int       initial level of schooling
share       not present in KW       int       share of agents with respective initial
                                              level of schooling
lagged      :math:`d_4(15)`         int       was in education last year

max         not present in KW       int       maximum level of schooling
=======     ===================     ======    ==========================

.. Warning::

    Again, there is a small difference between this setup and Keane and Wolpin (1997).
    There is no automatic change in sign for the costs. Thus, e.g. a \$1,000 tuition
    cost must be specified as -1000.

Note that in order to implement the model based on agents with different initial levels
of schooling the three integer values - start, share, and lagged - have to be specified
together as a block.

**HOME**

.. include:: occ_home_params.rst

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

In alignment to Keane and Wolpin (1994), the error terms of the model are set to follow
a multivariate normal distribution, allowing for cross-corellation are admissible, and
excluding serial corelation. In the initialization file, the shock parameters have to be
specified as standard deviations (single-digit subscipts) and covariances (double-digit
subscipt). In the implemetation, the requested number of realizations is drawn from the
standard normal distribution. The draws are then multiplied by the shock parameters set
in the initialization file in order to generate the desired variance-covariance
structure.

**TYPE SHARES**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
coeff       float       share of agents of type 2
coeff       float       effect of having aquired >10 years of schooling

coeff       float       share of agents of type 3
coeff       float       effect of having aquired >10 years of schooling
=======     ======      ==========================


**TYPE SHIFTS**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
coeff       float       deviation for Type 2 from Type 1 in occupation A contant
coeff       float       deviation for Type 2 from Type 1 in occupation B contant
coeff       float       deviation for Type 2 from Type 1 in education contant
coeff       float       deviation for Type 2 from Type 1 in home contant

coeff       float       deviation for Type 3 from Type 1 in occupation A contant
coeff       float       deviation for Type 3 from Type 1 in occupation B contant
coeff       float       deviation for Type 3 from Type 1 in education contant
coeff       float       deviation for Type 3 from Type 1 in home contant
=======     ======      ==========================

In this example initialization file the model implementation implies three types of
heterogenous agents. The current version of the code works both with more than three
types, as well as with homogenous agents (only one type). In order to add a type, a
block of two and a block of four coefficients need to be specified in the sections
``type shares`` and "T``type shifts``", respectively.

**SOLUTION**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
draws       int         number of draws for :math:`E\max`
store       bool        persistent storage of results
seed        int         random seed for :math:`E\max`
=======     ======      ==========================

**SIMULATION**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
agents      int         number of simulated agents
file        str         file to print simulated sample
seed        int         random seed for agent experience
=======     ======      ==========================

**ESTIMATION**

==========      ======      ==========================
Key             Value       Interpretation
==========      ======      ==========================
agents          int         number of agents to read from sample
draws           int         number of draws for choice probabilities
file            str         file to read observed sample
maxfun          int         maximum number of function evaluations
optimizer       str         optimizer to use
seed            int         random seed for choice probability
tau             float       scale parameter for function smoothing
==========      ======      ==========================

**DERIVATIVES**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
version     str         approximation scheme
=======     ======      ==========================

The computed derivatives are calculated numerically and are used in the standard error
calculation.

**PRECONDITIONING**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
eps         int         step size
minimum     int         minimum admissible value
type        str         preconditioning type
=======     ======      ==========================

The inputs in the Preconditioning block are employed in reaching a (faster) solution in
the optimization step. The coefficients are transformed for better handling by the
optimizer. Three different types of transformations can be selected via the
preconditioning type:

* identity - no transformation
* magnitude - divison by the number of digits
* gradient based - weighting by the inverse contribution to the likelihood function

**PROGRAM**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
debug       bool        debug mode
procs       int         number of processors
threads     int         number of threads
version     str         program version
=======     ======      ==========================

**INTERPOLATION**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
flag        bool        flag to use interpolation
points      int         number of interpolation points
=======     ======      ==========================

The implemented optimization algorithms vary with the program's version. If you request
the Python version of the program, you can choose from the ``scipy`` implementations of
the BFGS  (Norcedal and Wright, 2006), LBFGSB, and POWELL (Powell, 1964) algorithms. In
essense, POWELL is a conjugate direction method, which performs sequential
one-dimentional minimizations, does not require that the functions be differentiable and
no derivatives are taken. The BFGS algorythm is a quasi-Newton type of optimizer, which
uses first derivatives only, but performs reasonably well even in non-smooth
optimizations. The LBFGS algorithm can use simple box contraints to potentially improve
accuracy. Further implementation details are available `here
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`__.
For Fortran, we implemented the BFGS, BOBYQA and NEWUOA (Powell, 2004) algorithms.
NEWUOA is a gradient-free algorythm which performs unconstrained optimiztion. In a
similar fashion, BOBYQA performs gradient-free bound constrained optimization.

**FORT-NEWUOA**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
maxfun      float       maximum number of function evaluations
npt         int         number of points for approximation model
rhobeg      float       starting value for size of trust region
rhoend      float       minimum value of size for trust region
=======     ======      ==========================

**FORT-BFGS**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
eps         int         value to use for step size if fprime is approximated
gtol        float       gradient norm must be less than gtol before successful
                        termination
maxiter     int         maximum number of iterations
stpmx       int         maximum step size
=======     ======      ==========================


**FORT-BOBYQA**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
maxfun      float       maximum number of function evaluations
npt         int         number of points for approximation model
rhobeg      float       starting value for size of trust region
rhoend      float       minimum value of size for trust region
=======     ======      ==========================

**SCIPY-BFGS**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
eps                     value to use for step size if fprime is approximated
gtol        float       gradient norm must be less than gtol before successful
                        termination
maxiter     int         maximum number of iterations
stpmx       int         maximum step size
=======     ======      ==========================

**SCIPY-POWELL**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
ftol        float       relative error in func(xopt) acceptable for convergence
maxfun      int         maximum number of function evaluations to make
maxiter     int         maximum number of iterations
xtol        float       line-search error tolerance
=======     ======      ==========================

**SCIPY-LBFGSB**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
eps         float       Step size used when approx_grad is True, for numerically
                        calculating the gradient
factr       float       Multiple of the default machine precision used to determine the
                        relative error in func(xopt) acceptable for convergence
m           int         Maximum number of variable metric corrections used to define the
                        limited memory matrix.
maxiter     int         maximum number of iterations
maxls       int         Maximum number of line search steps (per iteration). Default is
                        20.
pgtol       float       gradient norm must be less than gtol before successful
                        termination
=======     ======      ==========================

Constraints for the Optimizer
-----------------------------

If you want to keep any parameter fixed at the value you specified (i.e. not estimate
this parameter) you can simply add an exclamation mark after the value. If you want to
provide bounds for a constrained optimizer you can specify a lower and upper bound in
round brackets. A section of such an .ini file would look as follows:

.. code-block::

    coeff             -0.049538516229344
    coeff              0.020000000000000     !
    coeff             -0.037283956168153       (-0.5807488086366478,None)
    coeff              0.036340835226155     ! (None,0.661243603948984)

In this example, the first coefficient is free. The second one is fixed at 0.2. The
third one will be estimated but has a lower bound. In the fourth case, the parameter is
fixed and the bounds will be ignored.

If you specify bounds for any free parameter, you have to choose a constraint optimizer
such as SCIPY-LBFGSB or FORT-BOBYQA.

Dataset
-------

To use respy, you need a dataset with the following columns:

- Identifier: identifies the different individuals in the sample
- Period: identifies the different rounds of observation for each individual
- Choice: an integer variable that indicates the labor market choice
    - 1 = Occupation A
    - 2 = Occupation B
    - 3 = Education
    - 4 = Home
- Earnings: a float variable that indicates how much people are earning. This variable
  is missing (indicated by a dot) if individuals don't work.
- Experience_A: labor market experience in sector A
- Experience_B: labor market experience in sector B
- Years_Schooling: years of schooling
- Lagged_Choice: choice in the period before the model starts. Codes are the same as in
  Choice.

The information in the data file should be first sorted by individual and then by period
as visualized below:

===     ======    ======      =========      ======    ======    =====    ===========
ID.     Priod     Choice      Earnings       Exp_A     Exp_B     sch_y    choice_lag
===     ======    ======      =========      ======    ======    =====    ===========
0       0         4           0              0         0         10       1
0       1         4           0              0         0         10       0
0       2         4           0              0         0         10       0
1       0         4           0              0         0         10       1
1       1         4           0              0         0         10       0
1       2         4           0              0         0         10       0
2       0         4           0              0         0         10       1
2       1         4           0              0         0         10       0
2       1         4           0              0         0         10       0
===     ======    ======      =========      ======    ======    =====    ===========

Datasets for respy are stored in simple text files, where columns are separated by
spaces. The easiest way to write such a text file in Python is to create a pandas
DataFrame with all relevant columns and then storing it in the following way:

.. code-block:: python

    with open("my_data.respy.dat", "w") as file:
        df.to_string(file, index=False, header=True, na_rep=".")

