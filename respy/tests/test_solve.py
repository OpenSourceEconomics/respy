import numpy as np
import pandas as pd
import pytest

from respy.config import EXAMPLE_MODELS
from respy.config import INDEXER_INVALID_INDEX
from respy.config import KEANE_WOLPIN_1994_MODELS
from respy.config import KEANE_WOLPIN_1997_MODELS
from respy.interface import get_example_model
from respy.pre_processing.model_checking import check_model_solution
from respy.pre_processing.model_processing import process_params_and_options
from respy.shared import create_core_state_space_columns
from respy.solve import get_solve_func
from respy.state_space import _create_core_period_choice
from respy.state_space import _create_core_state_space
from respy.state_space import _create_indexer
from respy.state_space import _insert_indices_of_child_states
from respy.state_space import create_state_space_class
from respy.tests._former_code import _create_state_space_kw94
from respy.tests._former_code import _create_state_space_kw97_base
from respy.tests._former_code import _create_state_space_kw97_extended
from respy.tests.random_model import generate_random_model
from respy.tests.utils import apply_to_attributes_of_two_state_spaces
from respy.tests.utils import process_model_or_seed


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS)
def test_check_solution(model_or_seed):

    params, options = process_model_or_seed(model_or_seed)

    solve = get_solve_func(params, options)
    state_space = solve(params)

    optim_paras, options = process_params_and_options(params, options)

    check_model_solution(optim_paras, options, state_space)


@pytest.mark.integration
@pytest.mark.precise
@pytest.mark.parametrize("model", EXAMPLE_MODELS)
def test_state_space_restrictions_by_traversing_forward(model):
    """Test for inadmissible states in the state space.

    The test is motivated by the addition of another restriction in
    https://github.com/OpenSourceEconomics/respy/pull/145. To ensure that similar errors
    do not happen again, this test takes all states of the first period and finds all
    their child states. Taking only the child states their children are found and so on.
    At last, the set of visited states is compared against the total set of states.

    The test can only applied to some models. Most models would need custom
    ``options["core_state_space_filters"]`` to remove inaccessible states from the state
    space.

    """
    params, options = process_model_or_seed(model)
    optim_paras, options = process_params_and_options(params, options)

    solve = get_solve_func(params, options)
    state_space = solve(params)

    out = dict()
    for x in state_space.child_indices.values():
        print(x)
        array = np.concatenate(x)
        for state in array:
            if state[0] in out.keys():
                if state[1] not in out[state[0]]:
                    out[state[0]].append(state[1])
                else:
                    continue
            else:
                out[state[0]] = [state[1]]

    for x in out:
        assert len(out[x]) == len(state_space.core_index_to_indices[x])


@pytest.mark.integration
@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS)
def test_invariance_of_solution(model_or_seed):
    """Test for the invariance of the solution.

    We run solve two times and check whether all attributes of the state space match.

    """
    params, options = process_model_or_seed(model_or_seed)

    optim_paras, options = process_params_and_options(params, options)

    solve = get_solve_func(params, options)
    state_space = solve(params)
    state_space_ = solve(params)

    apply_to_attributes_of_two_state_spaces(
        state_space.core, state_space_.core, np.testing.assert_array_equal
    )
    apply_to_attributes_of_two_state_spaces(
        state_space.get_attribute("wages"),
        state_space_.get_attribute("wages"),
        np.testing.assert_array_equal,
    )
    apply_to_attributes_of_two_state_spaces(
        state_space.get_attribute("nonpecs"),
        state_space_.get_attribute("nonpecs"),
        np.testing.assert_array_equal,
    )
    apply_to_attributes_of_two_state_spaces(
        state_space.get_attribute("expected_value_functions"),
        state_space_.get_attribute("expected_value_functions"),
        np.testing.assert_array_equal,
    )
    apply_to_attributes_of_two_state_spaces(
        state_space.get_attribute("base_draws_sol"),
        state_space_.get_attribute("base_draws_sol"),
        np.testing.assert_array_equal,
    )


@pytest.mark.precise
@pytest.mark.unit
@pytest.mark.parametrize("model", KEANE_WOLPIN_1994_MODELS)
def test_create_state_space_vs_specialized_kw94(model):
    point_constr = {"n_lagged_choices": 1, "observables": False}
    params, options = process_model_or_seed(model, point_constr=point_constr)

    optim_paras, options = process_params_and_options(params, options)

    # Create old state space arguments.
    n_periods = options["n_periods"]
    n_types = optim_paras["n_types"]
    edu_max = optim_paras["choices"]["edu"]["max"]
    edu_starts = np.array(list(optim_paras["choices"]["edu"]["start"]))

    # Get states and indexer from old state space.
    states_old, indexer_old = _create_state_space_kw94(
        n_periods, n_types, edu_starts, edu_max
    )
    if n_types == 1:
        states_old = states_old[:, :-1]
        for i, idx in enumerate(indexer_old):
            shape = idx.shape
            indexer_old[i] = idx.reshape(shape[:-2] + (-1,))

    states_new = _create_core_state_space(optim_paras, options)
    core_period_choice = _create_core_period_choice(states_new, optim_paras, options)

    # I think here we can get more elegant! Or is this the only way?
    core_index_to_complex = {i: k for i, k in enumerate(core_period_choice)}
    core_index_to_indices = {
        i: core_period_choice[core_index_to_complex[i]] for i in core_index_to_complex
    }

    # Create sp indexer
    indexer = _create_indexer(states_new, core_index_to_indices, optim_paras)

    # Compare the state spaces via sets as ordering changed in some cases.
    states_old_set = set(map(tuple, states_old))
    states_new_set = set(map(tuple, states_new.to_numpy()))
    assert states_old_set == states_new_set

    # Compare indexers via masks for valid indices.
    for period in range(n_periods):
        index_old_period = indexer_old[period]
        index_old_period = index_old_period != INDEXER_INVALID_INDEX
        index_old_period = np.nonzero(index_old_period)

        indices_old = [
            [period] + [index_old_period[x][i] for x in range(len(index_old_period))]
            for i in range(len(index_old_period[0]))
        ]

        for index in indices_old:
            assert tuple(index) in indexer.keys()

        for index in indexer.keys():
            print(index)
            if index[0] == period:
                assert list(index) in indices_old


@pytest.mark.precise
@pytest.mark.unit
@pytest.mark.parametrize("model", KEANE_WOLPIN_1997_MODELS)
def test_create_state_space_vs_specialized_kw97(model):
    params, options = process_model_or_seed(model)

    # Reduce runtime
    options["n_periods"] = 10 if options["n_periods"] > 10 else options["n_periods"]

    optim_paras, options = process_params_and_options(params, options)

    # Create old state space arguments.
    n_periods = options["n_periods"]
    n_types = optim_paras["n_types"]
    edu_max = optim_paras["choices"]["school"]["max"]
    edu_starts = np.array(list(optim_paras["choices"]["school"]["start"]))

    # Get states and indexer from old state space.
    if model == "kw_97_basic":
        states_old, indexer_old = _create_state_space_kw97_base(
            n_periods, n_types, edu_starts, edu_max
        )
    else:
        states_old, indexer_old = _create_state_space_kw97_extended(
            n_periods, n_types, edu_starts, edu_max
        )

    states_old = states_old[:, :-1]

    states_new = _create_core_state_space(optim_paras, options)

    core_period_choice = _create_core_period_choice(states_new, optim_paras, options)

    # I think here we can get more elegant! Or is this the only way?
    core_index_to_complex = {i: k for i, k in enumerate(core_period_choice)}
    core_index_to_indices = {
        i: core_period_choice[core_index_to_complex[i]] for i in core_index_to_complex
    }

    # Create sp indexer
    indexer = _create_indexer(states_new, core_index_to_indices, optim_paras)

    # Compare the state spaces via sets as ordering changed in some cases.
    states_old_set = set(map(tuple, states_old))
    states_new_set = set(map(tuple, states_new.to_numpy()))
    assert states_old_set == states_new_set

    # Compare indexers via masks for valid indices.
    for period in range(n_periods):
        index_old_period = indexer_old[period] != INDEXER_INVALID_INDEX
        index_old_period = np.nonzero(index_old_period)

        indices_old = [
            [period]
            + [index_old_period[x][i] for x in range(len(index_old_period) - 1)]
            for i in range(len(index_old_period[0]))
        ]

        for index in indexer.keys():

            if index[0] == period:
                print(index)
                assert list(index) in indices_old

        for index in indices_old:
            assert tuple(index) in indexer.keys()


@pytest.mark.edge_case
@pytest.mark.integration
def test_explicitly_nonpec_choice_rewards_of_kw_94_one():
    params, options = get_example_model("kw_94_one", with_data=False)

    solve = get_solve_func(params, options)
    state_space = solve(params)

    for arr in state_space.nonpecs.values():
        assert (arr[:, :2] == 0).all()
        assert (arr[:, -1] == 17_750).all()
        if arr.shape[1] == 4:
            np.isin(arr[:, 2], [0, -4_000, -400_000, -404_000]).all()


@pytest.mark.edge_case
@pytest.mark.integration
def test_explicitly_nonpec_choice_rewards_of_kw_94_two():
    params, options = get_example_model("kw_94_two", with_data=False)

    solve = get_solve_func(params, options)
    state_space = solve(params)

    for arr in state_space.nonpecs.values():
        assert (arr[:, :2] == 0).all()
        assert (arr[:, -1] == 14_500).all()
        if arr.shape[1] == 4:
            np.isin(arr[:, 2], [5_000, 0, -10_000, -15_000, -400_000, -415_000]).all()


def test_dense_choice_cores():
    """
    Check whether continuation values are equal for paths where
    the restrictions do not make
    any difference. We check continuation values
     at states where one choice leads to a remain
    ing decision tree that is equivalent to the unrestricted problem and one where this is not
    the case!
    """
    point_constr = {"n_periods": 6, "observables": [3]}

    params, options = generate_random_model(point_constr=point_constr)

    # Add some inadmissible states
    optim_paras, _ = process_params_and_options(params, options)

    # Solve the base model
    solve = get_solve_func(params, options)
    sp = solve(params)

    # Retrieve index
    edu_start = np.random.choice(list(optim_paras["choices"]["edu"]["start"].keys()))
    state = (3, 0, 3, edu_start, 1)
    core_ix = sp.indexer[state]

    # value = cont value at that point of the sp!
    pos = np.random.choice(range(len(sp.dense)))
    dense_combination = list(sp.dense.keys())[pos]
    dense_index = sp.dense_covariates_to_index[dense_combination]
    ix = (sp.core_to_index[core_ix[0], dense_index], core_ix[1])

    unrestricted_cont = sp.get_continuation_values(3)[ix[0]][ix[1]]

    # Impose some restriction
    options["inadmissible_states"] = {"a": ["period == 4 & exp_b ==4"]}

    # Solve the base model
    solve = get_solve_func(params, options)
    sp = solve(params)

    # Check some features of the sp
    restricted_cont = sp.get_continuation_values(3)[ix[0]][ix[1]]

    for i in [0, 2, 3]:
        assert restricted_cont[i] == unrestricted_cont[i]
    assert restricted_cont[1] != unrestricted_cont[1]


def test_invariance_of_wage_calc():
    """
    Simple
    """
    point_constr = {"n_periods": 2, "observables": [3]}

    params, options = generate_random_model(point_constr=point_constr)

    # Add some inadmissible states
    optim_paras, _ = process_params_and_options(params, options)

    # SOlve first model
    solve = get_solve_func(params, options)
    sp = solve(params)

    pos = np.random.choice(range(len(sp.dense)))
    dense_combination = list(sp.dense.keys())[pos]
    dense_index = sp.dense_covariates_to_index[dense_combination]
    idx = sp.core_to_index[(1, dense_index)]

    # Solve relevant wages
    wages_b = sp.wages[idx][:, 1]

    # Impose some restriction
    options["inadmissible_states"] = {"a": ["period == 1"]}

    solve = get_solve_func(params, options)
    sp = solve(params)

    wages_b_alt = sp.wages[idx][:, 0]

    np.testing.assert_array_equal(wages_b, wages_b_alt)


def test_child_indices():
    point_constr = {"n_periods": 2, "n_lagged_choices": 1}

    params, options = generate_random_model(point_constr=point_constr)

    # Add some inadmissible states
    optim_paras, options = process_params_and_options(params, options)

    sp = create_state_space_class(optim_paras, options)

    # Create all relevant columns
    core_columns = ["period"]
    core_columns += create_core_state_space_columns(optim_paras)

    # compose child indices of first choice
    initial_state = sp.core.iloc[0][core_columns].to_numpy()

    # Get all the future states
    states = list()
    for i, choice in enumerate(optim_paras["choices"]):
        child = initial_state.copy()
        child[0] += 1
        child[i + 1] += 1
        child[-1] = i
        print(child)
        ix = sp.indexer[(tuple(child))]
        states.append(np.array(ix).reshape(1, 2))

    manual = np.concatenate(states, axis=0)
    np.testing.assert_array_equal(sp.child_indices[0][0], manual)


def test_equality_of_equivalent_choice_sets():
    """
    Check for equcivalence
    """
    point_constr = {"n_periods": 3, "n_lagged_choices": 1}

    params, options = generate_random_model(point_constr=point_constr)

    optim_paras, options_ = process_params_and_options(params, options)
    sp = create_state_space_class(optim_paras, options_)

    choice = np.random.choice(list(optim_paras["choices_w_exp"].keys()))
    options["inadmissible_states"] = {
        choice: [
            f"period==2 & exp_{choice}==0",
            f"period==2 & exp_{choice}==1",
            f"period==2 & exp_{choice}==1",
        ]
    }

    optim_paras, options_ = process_params_and_options(params, options)
    sp_alt = create_state_space_class(optim_paras, options_)

    for i in sp_alt.index_to_indices:
        np.testing.assert_array_equal(
            sp_alt.index_to_indices[i], sp.index_to_indices[i]
        )


def test_wage_nonpecs():
    point_constr = {"n_periods": 3, "n_lagged_choices": 1, "observables": [2]}

    params, options = generate_random_model(point_constr=point_constr)

    solve = get_solve_func(params, options)
    sp = solve(params)
    wages = sp.wages
    nonpecs = sp.nonpecs

    optim_paras, _ = process_params_and_options(params, options)

    wage_choice = ("wage", np.random.choice(optim_paras["choices_w_wage"]))
    nonpec_choice = ("nonpec", np.random.choice(list(optim_paras["choices"])))

    # Change specs accordingly
    for reward in [wage_choice, nonpec_choice]:
        constant = params.loc[(f"{reward[0]}_{reward[1]}", "constant"), "value"]
        params = params.drop(index=[(f"{reward[0]}_{reward[1]}", "constant")])

        for obs in range(2):
            params.loc[
                (f"{reward[0]}_{reward[1]}", f"observable_0_{obs}"), "value"
            ] += constant

    solve = get_solve_func(params, options)
    sp = solve(params)
    wages_alt = sp.wages
    nonpecs_alt = sp.nonpecs

    for i in wages:
        np.testing.assert_array_almost_equal(wages[i], wages_alt[i])

    for i in nonpecs:
        np.testing.assert_array_almost_equal(nonpecs[i], nonpecs_alt[i])
