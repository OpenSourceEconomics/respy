.. _model-specification:

Model Specification
===================

In the following, we discuss the model specification in greater detail. In case the
model specification is used to simulate a data set, the data generation is based on the
chosen parameters. As soon as the estimation procedure is invoked, the values specified
in the model specification are used as starting values for the optimization.

The model is specified in two separate files as we differentiate between the parameters
of the model and other options. As an example we take the base model of Keane and Wolpin
(1997).


Parameter specification
-----------------------

The following table shows a parameter specification for respy. The first two columns,
``category`` and ``name``, can be used for indexing. ``para`` contains the parameter
value. ``fixed`` indicates whether the parameter is held constant during optimization.
``lower`` and ``upper`` indicate lower and upper bounds for the parameter which are used
in conjunction with a constrained optimizer. In this example the discount factor is
bounded between 0.7 and 1.0. ``comment`` contains a short description of the parameter.

.. csv-table:: kw_97_base.csv
    :file: ../../respy/tests/resources/kw_97_base.csv
    :header-rows: 1

In alignment to Keane and Wolpin (1994), the error terms of the model are set to follow
a multivariate normal distribution, allowing for cross-correlation are admissible, and
excluding serial correlation. In the parameter specification, the shock parameters have
to be specified as the lower triangular Cholesky factor of the covariance matrix. In the
implementation, the requested number of realizations is drawn from the standard normal
distribution. The draws are then multiplied by the shock parameters implied by the
Cholesky factor in order to generate the desired variance-covariance structure.

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
specification file in the ``yaml`` format.

.. literalinclude:: ../../respy/tests/resources/kw_97_base.yaml
    :language: yaml
    :name: kw_97_base.yaml

Covariates, state space filters and inadmissible states
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Part of ``respy``'s flexibility comes from using formulas to specify model components.
The following components rely on formulas which are passed to ``df.eval`` to customize
the model.

- covariates for the reward functions or type probabilities
- filters for the state space which delete impossible states
- restrictions which prevent access to inadmissible states

For an introduction to ``df.eval`` see the `official pandas documentation
<https://pandas.pydata.org/pandas-docs/stable/reference/api/
pandas.DataFrame.eval.html>`_ or this `post on StackOverflow
<https://stackoverflow.com/a/53779987>`_. Note the difference in implementations between
``df.eval`` (used here) and ``pd.eval``.

Helper functions
----------------

We provide some helper functions to write a model specification. You can use the
following function to output a template of the parameter specification.

.. autofunction:: respy.pre_processing.specification_helpers.csv_template

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
