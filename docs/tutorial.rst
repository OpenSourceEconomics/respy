Tutorial
========

We now illustrate the basic capabilities of the ``respy`` package. We start with the
model specification and then turn to some example use cases.

Model Specification
-------------------

The model is specified in an initialization file. Let us discuss each of its elements in
more detail.

**BASICS**

=======     ======      ==================
Key         Value       Interpretation
=======     ======      ==================
periods      int        number of periods
delta        float      discount factor
=======     ======      ==================

**COMMON**

=======     ======      ==================
Key         Value       Interpretation
=======     ======      ==================
coeff       float       return to hight school degree
coeff       float       return to college education
=======     ======      ==================

.. Warning::

    There are two small differences compared to Keane and Wolpin (1997). First, all
    coefficients enter the return function with a positive sign, while the squared terms
    enter with a minus in the original paper. Second, the order of covariates is fixed
    across the two occupations. In the original paper, own experience always comes
    before other experience.

**OCCUPATION A**

=======     ======    ==============
Key         Value     Interpretation
=======     ======    ==============
coeff       float     intercept
coeff       float     return to schooling
coeff       float     experience Occupation A, linear
coeff       float     experience Occupation A, squared
coeff       float     experience Occupation B, linear
coeff       float     experience Occupation B, squared
coeff       float     return to high school degree
coeff       float     return to college degree
coeff       float     linear return from growing older
coeff       float     effect of being a minor
coeff       float     gain from having worked in the occupation at least once before
coeff       float     gain from remaining in the same occupation as in previous period

coeff       float     constant
coeff       float     effect of experience in occupation A, but not from last period
coeff       float     effect of never before having worked in occupation A
=======     ======    ==============

**OCCUPATION B**

=======     ======    ================
Key         Value     Interpretation
=======     ======    ================
coeff       float     intercept
coeff       float     return to schooling
coeff       float     return to experience Occupation A, linear
coeff       float     return to experience Occupation A, squared
coeff       float     return to experience Occupation B, linear
coeff       float     return to experience Occupation B, squared
coeff       float     return to high school degree
coeff       float     return to college degree
coeff       float     linear return from growing older
coeff       float     effect of being a minor
coeff       float     gain from having worked in the occupation at least once before
coeff       float     gain from remaining in the same occupation as in previous period

coeff       float     constant
coeff       float     effect of experience in occupation B, but not from last period
coeff       float     effect of never before having worked in occupation B
=======     ======    ================

**EDUCATION**

======= ======    ==========================
Key     Value       Interpretation
======= ======    ==========================
coeff    float    constant, consumption value of school attendance
coeff    float    return to high school degree
coeff    float    return to college degree
coeff    float    cost of is_return_not_highschool
coeff    float    cost of is_return_high_school
coeff    float    linear return from growing older
coeff    float    effect of being a minor


start    int      initial level of schooling
share    int      share of agents with respective initial level of schoolong
lagged   int      was in education last year

start    int      initial level of schooling
share    int      share of agents with respective initial level of schoolong
lagged   int      was in education last year

max      int      maximum level of schooling
======= ======    ==========================

.. Warning::

    Again, there is a small difference between this setup and Keane and Wolpin (1997).
    There is no automatic change in sign for the costs. Thus, e.g. a \$1,000 tuition
    cost must be specified as -1000.

Note that in order to implement the model based on agents with different nitial levels
of schooling the three integer values - start, share, and lagged - have to be specified
together as a block.

**HOME**

======= ======      ==========================
Key     Value       Interpretation
======= ======      ==========================
coeff    float      mean value of non-market alternative
coeff    float      value if aged 18-20
coeff    float      value if 21 or older
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

**PRECONDITIONING**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
eps         int         step size
minimum     int         minimum admissible value
type        str         preconditioning type
=======     ======      ==========================

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
the BFGS  (Norcedal and Wright, 2006) and POWELL (Powell, 1964) algorithm. Their
implementation details are available `here <https://docs.scipy.org/doc/scipy-0.17.0/
reference/generated/scipy.optimize.minimize.html>`_. For Fortran, we implemented the
BFGS and NEWUOA (Powell, 2004) algorithms.

**FORT-NEWUOA**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
maxfun      int         maximum number of function evaluations
npt         int         number of points for approximation model
rhobeg      float       starting value for size of trust region
rhoend      float       minimum value of size for trust region
=======     ======      ==========================

**FORT-BFGS**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
eps         float       value to use for step size if fprime is approximated
gtol        float       gradient norm must be less than gtol before successful
                        termination
maxiter     int         maximum number of iterations
stpmx       float       maximum step size
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
eps         float       value to use for step size if fprime is approximated
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


Archive
-------
**PARALLELISM**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
flag        bool        parallel executable
procs       int         number of processors
=======     ======      ==========================


**SCALING**

=======     ======      ==========================
Key         Value       Interpretation
=======     ======      ==========================
flag        bool        apply scaling to parameters
minimum     float       minimum value for gradient approximation
=======     ======      ==========================




Constraints for the Optimizer
-----------------------------

If you want to keep any parameter fixed at the value you specified (i.e. not estimate
this parameter) you can simply add an exclamation mark after the value. If you want to
provide bounds for a constrained optimizer you can specify a lower and upper bound in
round brackets. A section of such an .ini file would look as follows::

    coeff             -0.049538516229344
    coeff              0.020000000000000     !
    coeff             -0.037283956168153       (-0.5807488086366478,None)
    coeff              0.036340835226155     ! (None,0.661243603948984)

In this example, the first coefficient is free. The second one is fixed at 0.2. The
third one will be estimated but has a lower bound. In the fourth case, the parameter is
fixed and the bounds will be ignored.

If you specify bounds for any free parameter, you have to choose a constraint optimizer
such as SCIPY-LBFGSB or FORT-BOBYQA.
