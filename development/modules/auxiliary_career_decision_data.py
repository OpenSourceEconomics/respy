"""This module contains the auxiliary function for the data transformation."""
import pandas as pd
import numpy as np


def prepare_dataset():
    """Convert the raw dataset to a DataFrame that can be used with Respy."""
    df = load_dataset()
    df = minor_refactoring(df)
    df = truncate_military_history(df)
    df = add_state_variables(df)
    write_out(df)


def load_dataset():
    """This function just prepares the original submission in a data frame."""
    columns = ["Identifier", "Age", "Schooling", "Choice", "Wage"]
    dtype = {
        "Identifier": np.int,
        "Age": np.int,
        "Schooling": np.int,
        "Choice": "category",
    }
    df = pd.DataFrame(
        np.genfromtxt("../../../respy/tests/resources/KW_97.raw"), columns=columns
    ).astype(dtype)

    df.set_index(["Identifier", "Age"], inplace=True, drop=False)

    # I drop the information on the NLSY identifier and set the identifier to the count.
    for count, idx in enumerate(df.index.levels[0]):
        df.loc[(slice(idx, idx), slice(None)), "Identifier"] = count
    df.set_index(["Identifier", "Age"], inplace=True, drop=False)

    # This mapping is based on direct communication with Kenneth Wolpin.
    df["Choice"].cat.categories = ["Schooling", "Home", "White", "Blue", "Military"]

    return df


def minor_refactoring(df):
    """This function performs some basic refactoring directly from existing variables."""

    df["Period"] = df["Age"] - 16
    df["Choice"].cat.categories = [3, 4, 1, 2, -99]
    df.rename(columns={"Schooling": "Years_Schooling"}, inplace=True)

    return df


def truncate_military_history(df):
    """This function truncates in individual's history once assigned to the military."""

    def _delete_military_service(agent):
        """This function deletes all observations going forward if an individual enrolls in the
        military."""
        for index, row in agent.iterrows():
            identifier, period = index
            if row["Choice"] == -99:
                return agent.loc[(slice(None, None), slice(None, period - 1)), :]
        return agent

    # pandas 23 does not allow for two index levels with the same name.
    # this was introduced on purpose, but currently breaks many sensible
    # groupby operations. This will probably be fixed in future versions
    # but here we introduce a workaround:
    # (the workaround should also work in all earlier and later versions.)
    df["id"] = df["Identifier"]
    df = df.groupby("id").apply(_delete_military_service)
    df.set_index(["Identifier", "Age"], inplace=True, drop=False)
    df.drop("id", axis=1, inplace=True)

    return df


def add_state_variables(df):
    """This function adds all additional state variables."""

    def _add_state_variables(agent):
        """This function iterates through an agent record and constructs the state variables for
        each point in tim.
        """
        exp_a, exp_b = 0, 0

        # We simply assume that individuals who do not have the expected number of years of
        # education did spend the last year at home.
        if agent.loc[(slice(None), slice(16, 16)), "Years_Schooling"].values < 10:
            lagged_choice = 4
        else:
            lagged_choice = 3

        for index, row in agent.iterrows():
            identifier, period = index

            agent["Lagged_Choice"].loc[:, period] = lagged_choice
            agent["Experience_A"].loc[:, period] = exp_a
            agent["Experience_B"].loc[:, period] = exp_b

            # Update labor market experience
            if row["Choice"] == 1:
                exp_a += 1
            elif row["Choice"] == 2:
                exp_b += 1
            else:
                pass

            # Update lagged activity:
            #   (0) Home, (1) Education, (2) Occupation A, and (3) Occupation B.
            lagged_choice = row["Choice"]

        return agent

    df["Lagged_Choice"] = np.nan
    df["Experience_A"] = np.nan
    df["Experience_B"] = np.nan

    # pandas 23 does not allow for two index levels with the same name.
    # this was introduced on purpose, but currently breaks many sensible
    # groupby operations. This will probably be fixed in future versions
    # but here we introduce a workaround:
    # (the workaround should also work in all earlier and later versions.)
    df["id"] = df["Identifier"]
    df = df.groupby("id").apply(_add_state_variables)
    df.drop("id", axis=1, inplace=True)

    return df


def write_out(df):
    labels = [
        "Identifier",
        "Period",
        "Choice",
        "Wage",
        "Experience_A",
        "Experience_B",
        "Years_Schooling",
        "Lagged_Choice",
    ]

    formats = {label: np.int for label in labels}
    formats["Wage"] = np.float

    """This function writes out the relevant information to a simple text file."""
    df = df[labels].astype(formats)
    with open("career_data.respy.dat", "w") as file_:
        df.to_string(file_, index=False, header=True, na_rep=".")

    df.set_index(["Identifier", "Period"], drop=False, inplace=True)
    df.to_pickle("career_data.respy.pkl")
