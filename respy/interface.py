from functools import partial

from respy.config import DATA_FORMATS_EST
from respy.config import DATA_LABELS_EST
from respy.likelihood import pyth_criterion
from respy.pre_processing.model_processing import parameters_to_vector
from respy.shared import create_multivariate_standard_normal_draws
from respy.simulate import simulate_data
from respy.solve import solve_with_backward_induction
from respy.solve import StateSpace


def minimal_solution_interface(attr, periods_draws_emax):

    state_space = StateSpace(
        attr["num_periods"],
        attr["num_types"],
        attr["edu_spec"]["start"],
        attr["edu_spec"]["max"],
        attr["optim_paras"],
    )

    state_space = solve_with_backward_induction(
        periods_draws_emax,
        state_space,
        attr["interpolation"],
        attr["num_points_interp"],
        attr["optim_paras"],
    )

    return state_space


def minimal_simulation_interface(attr):
    # Draw draws for the simulation.
    periods_draws_sims = create_multivariate_standard_normal_draws(
        attr["num_periods"], attr["num_agents_sim"], attr["seed_sim"]
    )

    # Draw standard normal deviates for the solution and evaluation step.
    periods_draws_emax = create_multivariate_standard_normal_draws(
        attr["num_periods"], attr["num_draws_emax"], attr["seed_emax"]
    )

    state_space = minimal_solution_interface(attr, periods_draws_emax)

    simulated_data = simulate_data(
        state_space,
        attr["num_agents_sim"],
        periods_draws_sims,
        attr["edu_spec"],
        attr["optim_paras"],
        attr["seed_sim"],
    )

    return state_space, simulated_data


def minimal_estimation_interface(attr, df):
    def get_independent_estimation_arguments(attr):
        periods_draws_prob = create_multivariate_standard_normal_draws(
            attr["num_periods"], attr["num_draws_prob"], attr["seed_prob"]
        )
        periods_draws_emax = create_multivariate_standard_normal_draws(
            attr["num_periods"], attr["num_draws_emax"], attr["seed_emax"]
        )

        # Construct the state space
        state_space = StateSpace(
            attr["num_periods"],
            attr["num_types"],
            attr["edu_spec"]["start"],
            attr["edu_spec"]["max"],
        )

        return periods_draws_emax, periods_draws_prob, state_space

    def get_parameter_vector(attr):
        """This function is obsolete with estimagic."""
        x = parameters_to_vector(
            attr["optim_paras"], attr["num_paras"], "all", attr["is_debug"]
        )

        return x

    def process_dataset(attr, df):
        """Process the dataset from disk."""

        # Process dataset from files.
        df.set_index(["Identifier", "Period"], drop=False, inplace=True)

        # We want to allow to estimate with only a subset of periods in the sample.
        cond = df["Period"] < attr["num_periods"]
        df = df[cond]

        # Only keep the information that is relevant for the estimation. Once that is
        # done,  impose some type restrictions.
        df = df[DATA_LABELS_EST]
        df = df.astype(DATA_FORMATS_EST)

        # We want to restrict the sample to meet the specified initial conditions.
        cond = df["Years_Schooling"].loc[:, 0].isin(attr["edu_spec"]["start"])
        df.set_index(["Identifier"], drop=False, inplace=True)
        df = df.loc[cond]

        # We now subset the dataframe to include only the number of agents that are
        # requested for the estimation. However, this requires to adjust the
        # num_agents_est as the dataset might actually be smaller as we restrict initial
        # conditions.
        df = df.loc[df.index.unique()[: attr["num_agents_est"]]]
        df.set_index(["Identifier", "Period"], drop=False, inplace=True)

        # We need to update the number of individuals for the estimation as the whole
        # dataset might actually be lower.
        num_agents_est = df["Identifier"].nunique()

        return df, num_agents_est

    # Process data
    df, attr["num_agents_est"] = process_dataset(attr, df)

    # Collect arguments for estimation.
    (
        periods_draws_emax,
        periods_draws_prob,
        state_space,
    ) = get_independent_estimation_arguments(attr)
    x = get_parameter_vector(attr)

    log_likelihood = partial(
        pyth_criterion,
        interpolation=attr["interpolation"],
        num_points_interp=attr["num_points_interp"],
        is_debug=attr["is_debug"],
        data=df,
        tau=attr["tau"],
        periods_draws_emax=periods_draws_emax,
        periods_draws_prob=periods_draws_prob,
        state_space=state_space,
    )

    return x, log_likelihood(x)
