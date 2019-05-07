.. _model-specification:

Model Specification
===================

In the following, we discuss the model specification in greater detail. In case the
model specification is used to simulate a data set, the data generation is based on the
chosen parameters. As soon as the estimation procedure is invoked, the values specified
in the model specification are used as starting values for the optimization.

The model is specified in two separate files as we differentiate between the parameters
of the model and other options. As an example we take the first parametrization of Keane
and Wolpin (1994).


Parameter specification
-----------------------

The following table shows a parameter specification for respy. The first two columns,
``category`` and ``name``, can be used for indexing. ``para`` contains the parameter
value. ``fixed`` indicates whether the parameter is held constant during optimization.
``lower`` and ``upper`` indicate lower and upper bounds for the parameter which are used
in conjunction with a constrained optimizer. In this example the discount factor is
bounded between 0.7 and 1.0. ``comment`` contains a short description of the parameter.

.. csv-table:: kw_data_one_types_initial.csv
    :file: ../respy/tests/resources/kw_data_one_types_initial.csv
    :header-rows: 1

In alignment to Keane and Wolpin (1994), the error terms of the model are set to follow
a multivariate normal distribution, allowing for cross-correlation are admissible, and
excluding serial correlation. In the initialization file, the shock parameters have to
be specified as standard deviations (single-digit subscripts) and covariances
(double-digit subscript). In the implementation, the requested number of realizations is
drawn from the standard normal distribution. The draws are then multiplied by the shock
parameters set in the initialization file in order to generate the desired
variance-covariance structure.

In this example specification the model implementation implies three types of
heterogeneous agents. The current version of the code works both with more than three
types, as well as with homogeneous agents (only one type). In order to add a type, a
block of two and a block of four coefficients need to be specified in the categories
``type_shares`` and ``type_shifts``", respectively.

.. Warning::

    There are two small differences compared to Keane and Wolpin (1997). First, all
    coefficients enter the return function with a positive sign, while the squared terms
    enter with a minus in the original paper. Second, the order of covariates is fixed
    across the two occupations. In the original paper, own experience always comes
    before other experience.

.. Warning::

    Again, there is a small difference between this setup and Keane and Wolpin (1997).
    There is no automatic change in sign for the costs. Thus, e.g. a \$1,000 tuition
    cost must be specified as -1000.


Options specification
---------------------

In addition to the model parameters, other model options are kept in another
specification file in the ``json`` format.

.. literalinclude:: ../respy/tests/resources/kw_data_one_types_initial.json
    :lines: 1-49, 62-67, 79-
    :language: json
    :name: kw_data_one_types_initial.json

Note that in order to implement the model based on agents with different initial levels
of schooling the three integer values - start, share, and lagged - have to be specified
together as a block.

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
