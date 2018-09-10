=====
TO-DO
=====

shared_auxiliary.py
===================

rewrite get_optim_paras to reduce hardcoding!



Questions:
----------

are dist_econ_paras and dist_optim_paras usually called with different types of parameter vectors? I think the difference is in the shock elements.




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



=====
Later
=====


Fine grained unit tests
=======================

Ideally we would have more fine grained unit tests for several functions. Since they only test small parts of code they will run quickly. For the same reason we can use higher precision. I propose tests for the following functions:

- pyth_contributions
- get_smoothed_probabilities
- all functions in solve
