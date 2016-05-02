.. _specification:

Model Specification
===================

We will now discuss the model. We first present the basic economic motivations and then show how to specify a particular model parameterization for the **respy** package.

Economics
---------

See the exposition in :ref:`Keane & Wolpin (1994) <bibSection>`.

Specification
-------------

The model is specified in an initialization file. For an example, check out the first specification analyzed in the original paper (Table 1) `online <https://github.com/restudToolbox/package/blob/master/example/data_one.ini>`_.

We turn to each of the ingredients in more details.

**BASICS**

=======     ======      ================== 
Key         Value       Interpretation      
=======     ======      ==================  
periods      int        number of periods  
delta        float      discount factor
=======     ======      ================== 

**OCCUPATION A**

=======     ======    ================== 
Key         Value       Interpretation      
=======     ======    ================== 
coeff       float       intercept  
coeff       float       return to schooling
coeff       float       experience occ. A, linear
coeff       float       experience occ. A, squared
coeff       float       experience occ. B, linear  
coeff       float       experience occ. B, squared
=======     ======    ================== 

There are two small differences between the setup in the original paper and the parameterization of the occupations. First, all coefficients enter the return function with a positive sign, while the squared terms enter with a minus in the original paper. Second, the order of covariates is fixed across the two occupations. In the original paper, own experience always comes before other experience. These comments are valid for Occupation A and Occupation B.

**OCCUPATION B**

=======     ======    ================== 
Key         Value       Interpretation      
=======     ======    ================== 
coeff       float       intercept  
coeff       float       return to schooling
coeff       float       experience occ. A, linear
coeff       float       experience occ. A, squared
coeff       float       experience occ. B, linear  
coeff       float       experience occ. B, squared
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

Again, there is a small difference between the setup in the original paper. There is no automatic change in sign for the tuition and adjustment costs. Thus, a \$4,000 tuition cost must be specified as -4,000

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
file        str         file to print simulated sample
agents      int         number of simulated agents
seed        int         random seed for agent experience
=======     ======      ==========================

**ESTIMATION**

==========      ======      ==========================
Key             Value       Interpretation      
==========      ======      ==========================
file            str         file to read observed sample
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
