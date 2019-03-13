import pandas as pd
import numpy as np
import copy
import os

from respy.python.shared.shared_auxiliary import get_conditional_probabilities
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.pre_processing.data_checking import check_estimation_dataset


def construct_transition_matrix(base_df):
    """ This method constructs the transition matrix.
    """
    df = base_df.copy(deep=True)
    df["Choice_Next"] = df.groupby(level="Identifier")["Choice"].shift(-1)
    args = []
    for label in ["Choice", "Choice_Next"]:
        args += [pd.Categorical(df[label], categories=range(1, 5))]
    tm = pd.crosstab(*args, normalize="index", dropna=False).values

    return tm


def get_final_education(agent):
    """ This method construct the final level of schooling for each individual.
    """
    edu_final = agent["Years_Schooling"].iloc[0] + (agent["Choice"] == 3).sum()

    # As a little test, we just ensure that the final level of education is equal or
    # less the level the agent entered the final period.
    valid = [agent["Years_Schooling"].iloc[-1], agent["Years_Schooling"].iloc[-1] + 1]
    np.testing.assert_equal(edu_final in valid, True)

    return edu_final


def write_info(respy_obj, data_frame):
    """ Write information about the simulated economy.
    """
    # Distribute class attributes
    optim_paras, num_types, file_sim, seed_sim, edu_spec = dist_class_attributes(
        respy_obj, "optim_paras", "num_types", "file_sim", "seed_sim", "edu_spec"
    )

    # Get basic information
    num_agents_sim = len(data_frame["Identifier"].unique())
    num_periods = len(data_frame["Period"].unique())

    # Write information to file
    with open(file_sim + ".respy.info", "w") as file_:

        file_.write("\n Simulated Economy\n\n")

        file_.write("   Number of Agents:       " + str(num_agents_sim) + "\n\n")
        file_.write("   Number of Periods:      " + str(num_periods) + "\n\n")
        file_.write("   Seed:                   " + str(seed_sim) + "\n\n\n")
        file_.write("   Choices\n\n")

        fmt_ = "{:>10}" + "{:>14}" * 4 + "\n\n"
        labels = ["Period", "Work A", "Work B", "School", "Home"]
        file_.write(fmt_.format(*labels))

        choices = data_frame["Choice"]
        for t in range(num_periods):
            args = []
            for decision in [1, 2, 3, 4]:
                args += [(choices.loc[slice(None), t] == decision).sum()]
            args = [x / float(num_agents_sim) for x in args]

            fmt_ = "{:>10}" + "{:14.4f}" * 4 + "\n"
            file_.write(fmt_.format((t + 1), *args))

        # We also print out the transition matrix as it provides some insights about the
        # persistence of choices. However, we can only compute this transition matrix if
        # the number of periods is larger than one.
        if num_periods > 1:
            file_.write("\n\n")
            file_.write("    Transition Matrix\n\n")
            fmt_ = "{:>10}" + "{:>14}" * 4 + "\n\n"
            labels = ["Work A", "Work B", "School", "Home"]
            file_.write(fmt_.format(*[""] + labels))

            tb = construct_transition_matrix(data_frame)
            for i in range(4):
                fmt_ = "    {:6}" + "{:14.4f}" * 4 + "\n"
                line = [labels[i]] + tb[i, :].tolist()

                # In contrast to the official documentation, the crosstab command omits
                # categories in the current pandas release when they are not part of the data. We
                # suspect this will be ironed out in the next releases.
                try:
                    file_.write(fmt_.format(*line))
                except IndexError:
                    pass

            file_.write("\n\n")

        # Now we can turn to the outcome information.
        file_.write("   Outcomes\n\n")

        for j, label in enumerate(["A", "B"]):

            file_.write("    Occupation " + label + "\n\n")
            fmt_ = "{:>10}" + "{:>14}" * 6 + "\n\n"

            labels = []
            labels += [" Period", "Counts", "Mean", "S.-Dev.", "2. Decile"]
            labels += ["5. Decile", "8. Decile"]

            file_.write(fmt_.format(*labels))

            for t in range(num_periods):

                is_working = choices.loc[slice(None), t] == j + 1
                wages = data_frame["Wage"].loc[slice(None), t][is_working]
                count = wages.count()

                if count > 0:
                    mean, sd = np.mean(wages), np.sqrt(np.var(wages))
                    percentiles = np.percentile(wages, [20, 50, 80]).tolist()
                else:
                    mean, sd = "---", "---"
                    percentiles = ["---", "---", "---"]

                values = [t + 1]
                values += [count, mean, sd]
                values += percentiles

                fmt_ = "{:>10}    " + "{:>10}    " * 6 + "\n"
                if count > 0:
                    fmt_ = "{:>10}    {:>10}" + "{:14.4f}" * 5 + "\n"
                file_.write(fmt_.format(*values))

            file_.write("\n")
        file_.write("\n")

        # Additional information about the simulated economy
        fmt_ = "    {:<16}" + "   {:15.5f}\n"
        file_.write("   Additional Information\n\n")

        stat = (data_frame["Choice"] == 1).sum() / float(num_agents_sim)
        file_.write(fmt_.format(*["Average Work A", stat]))

        stat = (data_frame["Choice"] == 2).sum() / float(num_agents_sim)
        file_.write(fmt_.format(*["Average Work B", stat]))

        # The calculation of years of schooling is a little more difficult to determine as we
        # need to account for the different levels of initial schooling. The column on
        # Years_Schooling only contains information on the level of schooling attainment going in
        # the period, thus is not identical to the final level of schooling for individuals that
        # enroll in school in the very last period.
        stat = data_frame.groupby(level="Identifier").apply(get_final_education).mean()
        file_.write(fmt_.format(*["Average School", stat]))

        stat = (data_frame["Choice"] == 4).sum() / float(num_agents_sim)
        file_.write(fmt_.format(*["Average Home", stat]))
        file_.write("\n")

        file_.write("\n\n   Schooling by Type\n\n")

        cat_schl = pd.Categorical(
            data_frame["Years_Schooling"][:, 0], categories=edu_spec["start"]
        )
        cat_type = pd.Categorical(data_frame["Type"][:, 0], categories=range(num_types))

        for normalize in ["all", "columns", "index"]:

            if normalize == "columns":
                file_.write("\n    ... by Type \n\n")
                num_columns = num_types + 1
            elif normalize == "index":
                file_.write("\n    ... by Schooling \n\n")
                num_columns = num_types
            else:
                file_.write("\n    ... jointly \n\n")
                num_columns = num_types

            info = pd.crosstab(
                cat_schl, cat_type, normalize=normalize, dropna=False, margins=True
            ).values

            fmt_ = "   {:>10}    " + "{:>25}" * num_columns + "\n\n"
            line = ["Schooling"]
            for i in range(num_types):
                line += ["Type " + str(i)]
            if num_columns == num_types + 1:
                line += ["All"]
            file_.write(fmt_.format(*line))

            fmt_ = "   {:>10}    " + "{:25.5f}" * num_columns + "\n"
            for i, start in enumerate(edu_spec["start"]):
                line = [start] + info[i, :].tolist()
                file_.write(fmt_.format(*line))

            if normalize == "index":
                fmt_ = "   {:>10}    " + "{:25.5f}" * num_columns + "\n"
                line = ["All"] + info[-1, :].tolist()
                file_.write(fmt_.format(*line))

        # We want to provide information on the value of the lagged activity when entering the
        # model based on the level of initial education.
        cat_1 = pd.Categorical(
            data_frame["Years_Schooling"][:, 0], categories=edu_spec["start"]
        )
        cat_2 = pd.Categorical(data_frame["Lagged_Choice"][:, 0], categories=[3, 4])
        info = pd.crosstab(cat_1, cat_2, normalize=normalize, dropna=False).values

        file_.write("\n\n   Initial Lagged Activity by Schooling\n\n")
        fmt_ = "\n   {:>10}" + "    {:>25}" + "{:>25}\n\n"
        file_.write(fmt_.format(*["Schooling", "Lagged Schooling", "Lagged Home"]))
        for i, edu_start in enumerate(edu_spec["start"]):
            fmt_ = "   {:>10}" + "    {:25.5f}" + "{:25.5f}\n"
            file_.write(fmt_.format(*[edu_start] + info[i].tolist()))

        file_.write("\n\n   Economic Parameters\n\n")
        fmt_ = "\n   {0:>10}" + "    {1:>25}\n\n"
        file_.write(fmt_.format(*["Identifier", "Value"]))
        vector = get_estimation_vector(optim_paras)
        fmt_ = "   {:>10}" + "    {:25.5f}\n"
        for i, stat in enumerate(vector):
            file_.write(fmt_.format(*[i, stat]))


def write_out(respy_obj, data_frame):
    """ Write dataset to file.
    """
    # Distribute class attributes
    file_sim = respy_obj.get_attr("file_sim")

    # We maintain several versions of the file.
    with open(file_sim + ".respy.dat", "w") as file_:
        data_frame.to_string(file_, index=False, header=True, na_rep=".")

    data_frame.to_pickle(file_sim + ".respy.pkl")


def format_float(x):
    """ Pretty formatting for floats
    """
    if pd.isnull(x):
        return "    ."
    else:
        return "{0:10.2f}".format(x)


def format_integer(x):
    """ Pretty formatting for integers.
    """
    if pd.isnull(x):
        return "    ."
    else:
        return "{0:<5}".format(int(x))


def get_estimation_vector(optim_paras):
    """ Construct the vector estimation arguments.
    """
    # Auxiliary objects
    num_types = int(len(optim_paras["type_shares"]) / 2)

    # Collect parameters
    vector = list()
    vector += optim_paras["delta"].tolist()
    vector += optim_paras["coeffs_a"].tolist()
    vector += optim_paras["coeffs_b"].tolist()
    vector += optim_paras["coeffs_edu"].tolist()
    vector += optim_paras["coeffs_home"].tolist()
    vector += optim_paras["shocks_cholesky"][0, :1].tolist()
    vector += optim_paras["shocks_cholesky"][1, :2].tolist()
    vector += optim_paras["shocks_cholesky"][2, :3].tolist()
    vector += optim_paras["shocks_cholesky"][3, :4].tolist()
    vector += optim_paras["type_shares"].tolist()

    for i in range(1, num_types):
        vector += optim_paras["type_shifts"][i, :].tolist()

    # Type conversion
    vector = np.array(vector)

    # Finishing
    return vector


def check_dataset_sim(data_frame, respy_obj):
    """ This routine runs some consistency checks on the simulated dataset.
    Some more restrictions are imposed on the simulated dataset than the
    observed data.
    """
    # Distribute class attributes
    num_agents = respy_obj.get_attr("num_agents_sim")
    num_periods = respy_obj.get_attr("num_periods")
    num_types = respy_obj.get_attr("num_types")

    # Some auxiliary functions for later
    def check_check_time_constant(group):
        np.testing.assert_equal(group["Type"].nunique(), 1)

    def check_number_periods(group):
        np.testing.assert_equal(group["Period"].count(), num_periods)

    # So, we run all checks on the observed dataset.
    check_estimation_dataset(data_frame, respy_obj)

    # Checks for PERIODS
    dat = data_frame["Period"]
    np.testing.assert_equal(dat.max(), num_periods - 1)

    # Checks for IDENTIFIER
    dat = data_frame["Identifier"]
    np.testing.assert_equal(dat.max(), num_agents - 1)

    # Checks for TYPES
    dat = data_frame["Type"]
    np.testing.assert_equal(dat.max() <= num_types - 1, True)
    np.testing.assert_equal(dat.isnull().any(), False)
    data_frame.groupby(level="Identifier").apply(check_check_time_constant)

    # Check that there are not missing wage observations if an agent is working. Also, we check
    # that if an agent is not working, there also is no wage observation.
    is_working = data_frame["Choice"].isin([1, 2])

    dat = data_frame["Wage"][is_working]
    np.testing.assert_equal(dat.isnull().any(), False)

    dat = data_frame["Wage"][~is_working]
    np.testing.assert_equal(dat.isnull().all(), True)

    # Check that there are no missing observations and we follow an agent each period.
    data_frame.groupby(level="Identifier").apply(check_number_periods)


def sort_type_info(optim_paras, num_types):
    """ We fix an order for the sampling of the types.
    """
    type_info = dict()

    # We simply fix the order by the size of the intercepts.
    type_info["order"] = np.argsort(optim_paras["type_shares"].tolist()[0::2])

    # We need to reorder the coefficients determining the type probabilities accordingly.
    type_shares = []
    for i in range(num_types):
        lower, upper = i * 2, (i + 1) * 2
        type_shares += [optim_paras["type_shares"][lower:upper].tolist()]
    type_info["shares"] = np.array(
        list(type_shares[i] for i in type_info["order"])
    ).flatten()

    return type_info


def sort_edu_spec(edu_spec):
    """ This function sorts the dictionary that provides the information about initial education.
    It adjusts the order of the shares accordingly.
    """
    edu_start_ordered = sorted(edu_spec["start"])

    edu_share_ordered = []
    for start in edu_start_ordered:
        idx = edu_spec["start"].index(start)
        edu_share_ordered += [edu_spec["share"][idx]]

    edu_spec_ordered = copy.deepcopy(edu_spec)
    edu_spec_ordered["start"] = edu_start_ordered
    edu_spec_ordered["share"] = edu_share_ordered

    return edu_spec_ordered


def get_random_types(num_types, optim_paras, num_agents_sim, edu_start, is_debug):
    """ This function provides random draws for the types, or reads them in from a file.
    """
    # We want to ensure that the order of types in the initialization file does not matter for
    # the simulated sample.
    type_info = sort_type_info(optim_paras, num_types)

    if is_debug and os.path.exists(".types.respy.test"):
        types = np.genfromtxt(".types.respy.test")
    else:
        types = []
        for i in range(num_agents_sim):
            probs = get_conditional_probabilities(type_info["shares"], edu_start[i])
            types += np.random.choice(type_info["order"], p=probs, size=1).tolist()

    # If we only have one individual, we need to ensure that types are a vector.
    types = np.array(types, ndmin=1)

    return types


def get_random_edu_start(edu_spec, num_agents_sim, is_debug):
    """ This function provides random draws for the initial schooling level, or reads
    them in from a file.
    """
    # We want to ensure that the order of initial schooling levels in the initialization
    # files does not matter for the simulated sample. That is why we create an ordered
    # version for this function.
    edu_spec_ordered = sort_edu_spec(edu_spec)

    if is_debug and os.path.exists(".initial_schooling.respy.test"):
        edu_start = np.genfromtxt(".initial_schooling.respy.test")
    else:
        # As we do not want to be too strict at the user-level the sum of edu_spec might
        # be slightly larger than one. This needs to be corrected here.
        probs = edu_spec_ordered["share"] / np.sum(edu_spec_ordered["share"])
        edu_start = np.random.choice(
            edu_spec_ordered["start"], p=probs, size=num_agents_sim
        )

    # If we only have one individual, we need to ensure that types are a vector.
    edu_start = np.array(edu_start, ndmin=1)

    return edu_start


def get_random_choice_lagged_start(edu_spec, num_agents_sim, edu_start, is_debug):
    """ This function provides values for the initial lagged choice.

    The values are random draws or read in from a file.

    """
    if is_debug and os.path.exists(".initial_lagged.respy.test"):
        lagged_start = np.genfromtxt(".initial_lagged.respy.test")
    else:
        lagged_start = []
        for i in range(num_agents_sim):
            idx = edu_spec["start"].index(edu_start[i])
            probs = edu_spec["lagged"][idx], 1 - edu_spec["lagged"][idx]
            lagged_start += np.random.choice([3, 4], p=probs, size=1).tolist()

    # If we only have one individual, we need to ensure that activities are a vector.
    lagged_start = np.array(lagged_start, ndmin=1)

    return lagged_start
