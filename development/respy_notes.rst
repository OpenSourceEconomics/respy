=====
TO-DO
=====

- update tutorial notebook


Interface
---------

- Change the user interface: estimate and simulate become methods of clsRespy. Estimate is called fit.

- shared_auxiliary.py
---------------------

- dist_econ_paras and dist_optim_paras shares most of the logic and just has different return types. It is not clear from the names what the difference is. I would suggest one public function with a switch (target='dict'; target='tuple') and potentially two private functions for the implementation.
- get_optim_paras should be closer to the previous two functions and not hardcode the parsing information.
    -> do both; why do we have the cholesky of covs at one place and not the other?


===============================
Notes and questions about respy
===============================

clsRespy
========


- update_optim paras could be made a lot shorter after my proposed changes to dist_econ_paras and dist_optim_paras; In any case it has an unnecessary assignment (self.attr['optim_paras'] = optim_paras) -> do it

model_processing_auxiliary.py
=============================

- Even though the _paras_mapping does have a pretty long docstring I don't understand it well enough to judge whether it is correct. -> Philipp checks that

estimate_python.py
==================

- pyth_criterion should be renamed to reflect that fact that it is a likelihood function. nloglike is a good name from statsmodels. -> do it


estimate_wrapper.py
===================

_construct_all_current_values needs a longer docstring. What is the purpose of this function?
    -> I understand now; put a section in the developer documentation.

evaluate_python
===============

- The dimensions and meaning of some arguments is still unclear
- Why is the covariance matrix never used. Is the use of the cholesky factors correct?


Reduce Fortran Code
===================

There are already some parts that are only implemented in Python and not Fortran or that are implemented very differently in Python. I think we should carve out everything that is not very speed relevant and only implement it in Python. Then we can use much more idiomatic Python in that case.

We should put all of this code into one folder called model_processing. Then we could write about the reasons for this directory structure in the documentation. This would make it much easier for new developers!

Already Python only
-------------------

- clsRespy
- estimate
- simulate
- process
- read


Proposed Python only
--------------------

- create_state_space -> save it to something that can be read by Fortran (probably not)
- create_draws -> save it to something that can be read by Fortran
- ?


Directory Structure
===================

I would summarize 'evaluate' and 'estimate' to 'likelihood'
I would summarize 'read' and 'process' to 'model_processing'


Control flow of the estimation
==============================

Current
-------

estimate calls interface(request='estimate')

interface creates arguments and does preconditioning

interface calls OptimizationClass

OptimizationClass calls pyth_criterion

we should look for a better version.



document the different types of the parameter vector and write one module where all transformatinos happen.

- x: array of model parameters
- x_all: including the fixed ones
- x_optim: only free parameters

    -> try to reduce number of representations of parameter vector



=====
Later
=====


Fine grained unit tests
=======================

Ideally we would have more fine grained unit tests for several functions. Since they only test small parts of code they will run quickly. For the same reason we can use higher precision. I propose tests for the following functions:

- pyth_contributions
- get_smoothed_probabilities
- all functions in solve
