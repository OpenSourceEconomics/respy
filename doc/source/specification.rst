.. _specification:

Basic Model
===========

We will now discuss the model. We first present the basic economic motivations and then show how to specifcy a particular model parameterization for the **respy** package.

Economics
---------

See the discussion in :ref:`Keane & Wolpin (1994) <bibSection>`.

Specification
-------------

The model is specified in an initialization file. For example, see `here <https://github.com/restudToolbox/package/blob/master/example/data_one.ini>`_ for the specification of the first dataset in the original paper (see Table 1).

We now discuss each of the ingredients in more details.

**BASICS**

=======  ================== 
Key      Interpretation      
=======  ==================  
periods  number of Periods  
delta    discount factor
=======  ================== 

**OCCUPATION A**

=======  ======================== 
Key      Interpretation      
=======  ======================== 
coeff    intercept  
coeff    return to schooling
coeff    experience occ. A, linear
coeff    experience occ. A, squared
coeff    experience occ. B, linear  
coeff    experience occ. B, squared
=======  ======================== 

There are two small differences between the setup in the original paper and the parameterization of the occupations. First, all coefficients enter the return function with a positive sign, while the squared terms enter with a minus in the original paper. Second, the order of covariates is fixed across the two occupations. In the original paper, own experience always comes before other experience. These comments are valid for Occupation A and Occupation B.

**OCCUPATION B**

=======  ======================== 
Key      Interpretation      
=======  ======================== 
coeff    intercept  
coeff    return to schooling
coeff    experience occ. A, linear
coeff    experience occ. A, squared
coeff    experience occ. B, linear  
coeff    experience occ. B, squared
=======  ======================== 

**EDUCATION**

=======  ==========================
Key      Interpretation      
=======  ========================== 
coeff    consumption value
coeff    tuition cost
coeff    adjustment cost
max      maximum level of schooling
start    initial level of schooling
=======  ========================== 

Again, there is a small difference between the setup in the original paper. There is no automatic change in sign for the tuition and adjustment costs. Thus, a \$4,000 tuition cost must be specified as -4,000

**HOME**

=======  ==========================
Key      Interpretation      
=======  ========================== 
coeff    mean value of non-market alternative
=======  ========================== 

**SHOCKS**

=======  ==========================
Key      Interpretation      
=======  ========================== 
coeff    :math:`\sigma_{1}`
coeff    :math:`\sigma_{12}`
coeff    :math:`\sigma_{13}`
coeff    :math:`\sigma_{14}`
coeff    :math:`\sigma_{2}`
coeff    :math:`\sigma_{23}`
coeff    :math:`\sigma_{24}`
coeff    :math:`\sigma_{3}`
coeff    :math:`\sigma_{34}`
coeff    :math:`\sigma_{4}`
=======  ========================== 

**SOLUTION**

=======  ==========================
Key      Interpretation      
=======  ========================== 
draws    number of draws for EMAX approximation
store    store results
seed     random seed for the EMAX approximation
=======  ========================== 

**SIMULATION**

=======  ==========================
Key      Interpretation      
=======  ========================== 
file     file to print simulated sample
agents   number of simulated agents
seed     random seed for agent experience
=======  ========================== 


**ESTIMATION**

=========  ==========================
Key        Interpretation      
=========  ==========================
file       file to read observed sample
tau        smoothing window
agents     number of agents to read from sample
maxiter    maximum number of iterations for optimizer
seed       random seed for choice probability approximation
optimizer  optimizer to use
=========  ==========================

**PROGRAM**

=======     ==========================
Key         Interpretation      
=======     ========================== 
debug       flag to use debug mode
version     program version
=======     ========================== 


**INTERPOLATION**

=======     ==========================
Key         Interpretation      
=======     ========================== 
points      number of interpolation points
apply       flag to use interpolation
=======     ========================== 

Two alternative optimization algorithms are available for the estimation. In both cases, we use the **scipy** package, see their documentation for more details.

**SCIPY-BFGS**

=======     ==========================
Key         Interpretation      
=======     ========================== 
gtol        gradient norm must be less than gtol before successful termination 
epsilon     step size for numerical approximation of first derivatives 
=======     ========================== 

**SCIPY-POWELL**

=======     ==========================
Key         Interpretation      
=======     ========================== 
maxfun      maximum number of function evaluations to make
ftol        relative error in func(xopt) acceptable for convergence      
xtol        line-search error tolerance         
=======     ========================== 
