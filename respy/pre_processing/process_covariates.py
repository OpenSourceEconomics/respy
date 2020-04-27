"""This module comprises all functions which process the definition of covariates."""
import copy


def remove_irrelevant_covariates(options, params):
    """Identify the relevant covariates.

    We try to make every model as sparse as possible which means discarding covariates
    which are irrelevant. The immediate benefit is that memory consumption and start-up
    costs are reduced.

    An advantage further downstream is that the number of lagged choices is inferred
    from covariates. Eliminating irrelevant covariates might reduce the number of
    implemented lags.

    The function catches all relevant "high-level" covariates by looking at the `"name"`
    index in `params`. "Low-level" covariates which are relevant but not included in the
    index are recursively found by checking whether covariates are used in the formula
    of relevant covariates.

    See also
    --------
    separate_covariates_into_core_dense_mixed

    """
    options = copy.deepcopy(options)
    covariates = options["covariates"]

    # Collect initial relevant covariates from params.
    relevant_covs = {}
    for cov in covariates:
        if cov in params.index.get_level_values("name"):
            relevant_covs[cov] = covariates[cov]

    # Start by iterating over initial covariates and add variables which are used to
    # compute them and repeat the process.
    n_relevant_covariates_changed = True
    while n_relevant_covariates_changed:
        n_relevant_covariates = len(relevant_covs)

        for cov in covariates:
            for relevant_cov in relevant_covs:
                if cov in relevant_covs[relevant_cov]:
                    # Append the covariate to the front such that nested covariates are
                    # created in the beginning.
                    relevant_covs = {cov: covariates[cov], **relevant_covs}

        n_relevant_covariates_changed = n_relevant_covariates != len(relevant_covs)

    options["covariates"] = relevant_covs

    return options


def separate_covariates_into_core_dense_mixed(options, optim_paras):
    """Separate covariates into distinct groups.

    Covariates are separated into three groups.

    1. Covariates which use only information from the core state space.
    2. Covariates which use only information from the dense state space.
    3. Covariates which use information from the core and the dense state space.

    Parameters
    ----------
    options : dict
        Contains among other information covariates and their formulas.
    optim_paras : dict
        Contains information to separate the core and dense state space.

    Returns
    -------
    options : dict
        Contains three new covariate categories.

    """
    options = copy.deepcopy(options)
    covariates = options["covariates"]

    # Define two sets with default covariates for the core and dense state space.
    core_covs = set(
        ["period"]
        + [f"exp_{choice}" for choice in optim_paras["choices_w_exp"]]
        + [f"lagged_choice_{i}" for i in range(1, optim_paras["n_lagged_choices"] + 1)]
    )
    dense_covs = set(optim_paras["observables"])
    if optim_paras["n_types"] >= 2:
        dense_covs |= set(
            ["type"] + [f"type_{i}" for i in range(2, optim_paras["n_types"] + 1)]
        )

    detailed_covariates = {
        cov: {"formula": covariates[cov], "depends_on": set()} for cov in covariates
    }

    # Loop over all covariates and add them two the sets if the formula contains
    # covariates from the sets. If both lengths of the sets do not change anymore, stop.
    n_core_covs_changed = True
    n_dense_covs_changed = True
    while n_core_covs_changed or n_dense_covs_changed:
        n_core_covs = len(core_covs)
        n_dense_covs = len(dense_covs)

        for cov, formula in covariates.items():
            matches_core = [i for i in core_covs if i in formula]
            if matches_core:
                core_covs.update([cov])

            matches_dense = [i for i in dense_covs if i in formula]
            if matches_dense:
                dense_covs.update([cov])

            detailed_covariates[cov]["depends_on"] |= set(matches_core + matches_dense)

        n_core_covs_changed = n_core_covs != len(core_covs)
        n_dense_covs_changed = n_dense_covs != len(dense_covs)

    only_core_covs = core_covs - dense_covs
    only_dense_covs = dense_covs - core_covs
    independent_covs = set(covariates) - core_covs - dense_covs

    options["covariates_core"] = {
        cov: detailed_covariates[cov]
        for cov in only_core_covs | independent_covs
        if cov in detailed_covariates
    }
    options["covariates_dense"] = {
        cov: detailed_covariates[cov]
        for cov in only_dense_covs
        if cov in detailed_covariates
    }
    options["covariates_mixed"] = {
        cov: detailed_covariates[cov] for cov in core_covs & dense_covs
    }
    # We cannot overwrite `options["covariates"]`.
    options["covariates_all"] = detailed_covariates

    return options


def separate_choice_restrictions_into_core_dense_mixed(options, optim_paras):
    """
    TODO: PRELIMINARY I AM NOT YET SURE WHETHER THIS LOGIC
          WORKS!
    """
    options = copy.deepcopy(options)

    # We want to build lists with all terms that are dense vars and all terms that are core
    # I am not sure whether this list is exhaustive and I can imagine there is a more elegant way to do so!
    # Dense
    dense_var = list(options["covariates_dense"].keys())
    dense_var = dense_var + list(optim_paras["observables"].keys())

    # Core
    core_var = list(options["covariates_core"].keys())
    core_var = core_var + [f"exp_{x}" for x in optim_paras["choices_w_exp"]]
    core_var = core_var + [
        f"lagged_choice_{x}" for x in range(1, optim_paras["n_lagged_choices"] + 1)
    ]
    core_var = core_var + ["period"]

    # Add ne dict keys
    constr_list = list()
    for choice in options["inadmissible_choices"].keys():
        for choice_constr in options["inadmissible_choices"][choice]:
            if any([x in choice_constr for x in dense_var]) == False:
                constr_list.append((choice_constr, choice, "core"))
            elif any([x in choice_constr for x in core_var]) == False:
                constr_list.append((choice_constr, choice, "dense"))
            else:
                constr_list.append((choice_constr, choice, "mixed"))

    for sp in ["core", "dense", "mixed"]:
        options[f"inadmissible_choices_{sp}"] = {}
        for choice in options["inadmissible_choices"].keys():
            relevant_contraints = [
                x for x in constr_list if x[1] == choice and x[2] == sp
            ]
            if relevant_contraints == []:
                pass
            else:
                options[f"inadmissible_choices_{sp}"][choice] = relevant_contraints
    return options


def identify_necessary_covariates(dependents, definitions):
    """Identify covariates necessary to compute `dependents`.

    This function can be used if only a specific subset of covariates is necessary and
    not all covariates.

    See also
    --------
    respy.likelihood._compute_x_beta_for_type_probability

    """
    dependents = {dependents} if isinstance(dependents, str) else set(dependents)
    new_dependents = dependents.copy()

    while new_dependents:
        deps = list(new_dependents)
        new_dependents = set()
        for dependent in deps:
            if dependent in definitions and definitions[dependent]["depends_on"]:
                dependents |= definitions[dependent]["depends_on"]
                new_dependents |= definitions[dependent]["depends_on"]
            else:
                dependents.remove(dependent)

    covariates = {dep: definitions[dep] for dep in dependents}

    return covariates
