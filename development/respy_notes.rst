======================================
Notes, todos and questions about respy
======================================


clsRespy
========


- Having global variables with capital letters is un-pythonic. Can we just make DERIVED_ATTR, SOLUTION_ATTR class attributes of clsRespy or are they used somewhere else? If so, shouldn't they have the same treatment as shared_constants? -> do it

- should we delete the init_dict at the end of __init__? It might become outdated. -> do it

- Even though the PARAS_MAPPING does have a pretty long comment I don't understand it well enough to judge whether it is correct. -> Philipp checks that

- What is tau? t> smoothing parameter

- update_optim paras could be made a lot shorter after my proposed changes to dist_econ_paras and dist_optim_paras; In any case it has an unnecessary assignment (self.attr['optim_paras'] = optim_paras) -> do it


- _initialize_attributes and update_core_attributes should probably be just one step. It could be a trivial one after settle on one dictionary representation. -> do it

- the distinction between update core_attributes and update_derived_attributes is no longer clear cut since num_paras is now a derived attribute! -> put num_paras in derived attributes

- in check_integrity_attributes we could save the unpacking step and instead write a = self.attr for shortcut access to the attributes. -> do it

- maybe replace attr by __dict__ -> do it


estimate.py and simulate.py
===========================

- I don't understand this comment:
    # Make sure all optimizers are fully defined for the FORTRAN interface.
    # At the same time, we do not want to require the user to specify only
    # the optimizers that are used. So, we sample a full set and replace the
    # optimizers that are used with the user specification.


- I think check_optimizer_options should be part of clsRespy. It might even be good to call that in the __init__ method. Moreover, I would make it optional to specify any optimizer options, so people who only simulate are not bothered by this check.
    - do it.


process_python.py
=================

- Process is not a very good name for the main function in this module. I would call the module data_processing, and the main function process_data. -> do it


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


read_python
===========

- The dict that comes out here should be already the attr dict, so we can just write self.attr.update(dict_) in clsRespy; Even better: maybe the whole function can be replaced by a call to json.load or yaml.load. -> make proposal


shared_auxiliary.py
===================

- where do we use the results of print_init_dict? Are they used for fortran? If not, the function can again be replaced by a simple call to json.dump or yaml.dump if we change from .ini files to dictionaries. -> do it


========
Results:
========

Interface
=========

- We should change the user interface in one of the following directions.
    1) estimate and simulate become methods of clsRespy. Estimate is called fit. This would be conceptually compatible with statsmodels and sklearn.

- Make a proposal for a yaml file that specifies the keane Wolpin model


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




shared_auxiliary.py
===================


- dist_econ_paras and dist_optim_paras shares most of the logic and just has different return types. It is not clear from the names what the difference is. I would suggest one public function with a switch (target='dict'; target='tuple') and potentially two private functions for the implementation.
- get_optim_paras should be closer to the previous two functions and not hardcode the parsing information.
    -> do both; why do we have the cholesky of covs at one place and not the other?





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
