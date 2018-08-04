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


shared_auxiliary.py
===================


- dist_econ_paras and dist_optim_paras shares most of the logic and just has different return types. It is not clear from the names what the difference is. I would suggest one public function with a switch (target='dict'; target='tuple') and potentially two private functions for the implementation.








