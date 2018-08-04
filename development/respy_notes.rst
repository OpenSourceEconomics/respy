======================================
Notes, todos and questions about respy
======================================


clsRespy
========


- Having global variables with capital letters is un-pythonic. Can we just make DERIVED_ATTR, SOLUTION_ATTR class attributes of clsRespy or are they used somewhere else? If so, shouldn't they have the same treatment as shared_constants?

- should we delete the init_dict at the end of __init__? It might become outdated.

- Even though the PARAS_MAPPING does have a pretty long comment I don't understand it well enough to judge whether it is correct.

- What is tau?

- update_optim paras could be made a lot shorter after my proposed changes to dist_econ_paras and dist_optim_paras; In any case it has an unnecessary assignment (self.attr['optim_paras'] = optim_paras)

- Many things (write_out, _udpate_core_attributes, ...) could be greatly simplified if we harmonize the different representations of a model definition. Currently we have: 1) init file, 2) init_dict (read from init file), 3) attr dict (updated from init_dict), but also contains derived attributes, 4) resfort_initialization (defined in respy/fortran/interface.py). My preferred solution would be to have one dictionary (fortran derived type) representation that is used throughout the package. A user would create an instance of clsRespy either directly with that dictionary or with a file_path to a json file. Having the option to create an instance from a dict would make large parts of the testing easier. write_out would just be a call to json.dump instead of a long function. Alternatively we could still require the user to write ini files and then convert them to the dictionary representation internally during the read step.

- _initialize_attributes and update_core_attributes should probably be just one step. It could be a trivial one after settle on one dictionary representation.

- the distinction between update core_attributes and update_derived_attributes is no longer clear cut since num_paras is now a derived attribute!

- in check_integrity_attributes we could save the unpacking step and instead write a = self.attr for shortcut access to the attributes.


estimate.py and simulate.py
===========================

- I don't understand this comment:
    # Make sure all optimizers are fully defined for the FORTRAN interface.
    # At the same time, we do not want to require the user to specify only
    # the optimizers that are used. So, we sample a full set and replace the
    # optimizers that are used with the user specification.

- I we should change the user interface in one of the following directions.
    1) estimate and simulate become methods of clsRespy. This would be conceptually compatible with statsmodels and sklearn.
    2) a user never explicitly makes an instance of clsRespy. Instead he just calls estimate or simulate with an init file. Stata people would like this one.

- I think check_optimizer_options should be part of clsRespy. It might even be good to call that in the __init__ method. Moreover, I would make it optional to specify any optimizer options, so people who only simulate are not bothered by this check.


process_python.py
=================

- Process is not a very good name for the main function in this module. I would call the module data_processing, and the main function process_data.






shared_auxiliary.py
===================


- dist_econ_paras and dist_optim_paras shares most of the logic and just has different return types. It is not clear from the names what the difference is. I would suggest one public function with a switch (target='dict'; target='tuple') and potentially two private functions for the implementation.



Control flow of the estimation
==============================

Current
-------

estimate calls interface(request='estimate')

interface creates arguments and toes preconditioning

interface calls OptimizationClass

OptimizationClass calls pyth_criterion

we should look for a better version.











