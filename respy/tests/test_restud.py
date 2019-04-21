import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import assert_frame_equal

import respy
from respy import RespyCls
from respy.pre_processing.model_processing import _create_attribute_dictionary
from respy.pre_processing.model_processing import _options_spec_from_attributes
from respy.pre_processing.model_processing import _params_spec_from_attributes
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_constants import IS_FORTRAN
from respy.python.shared.shared_constants import TEST_RESOURCES_BUILD
from respy.python.shared.shared_constants import TEST_RESOURCES_DIR
from respy.tests.codes.auxiliary import simulate_observed
from respy.tests.codes.random_model import generate_random_model


COLUMNS = []
COLUMNS += [
    "Identifier",
    "Total_Periods",
    "Choice",
    "Reward",
    "Experience_A",
    "Experience_B",
]
COLUMNS += ["Years_Schooling", "Schooling_Lagged"]

COLUMN_TYPES = {
    "Identifier": np.int,
    "Total_Periods": np.int,
    "Choice": np.int,
    "Reward": np.float,
    "Experience_A": np.float,
    "Experience_B": np.float,
    "Years_Schooling": np.int,
    "Schooling_Lagged": np.int,
}

DATA_LABELS_EST = []
DATA_LABELS_EST += [
    "Identifier",
    "Period",
    "Choice",
    "Wage",
    "Experience_A",
    "Experience_B",
]
DATA_LABELS_EST += ["Years_Schooling", "Lagged_Choice"]

DATA_FORMATS_EST = {}
for key_ in DATA_LABELS_EST:
    DATA_FORMATS_EST[key_] = np.int
    if key_ in ["Wage"]:
        DATA_FORMATS_EST[key_] = np.float


def restud_sample_to_respy():
    """This function transforms the RESTUD simulation sample for processing for the
    RESPY package."""

    def _add_period(agent):
        """This function adds the period information."""
        num_periods = int(agent["Total_Periods"].max())
        agent["Period"] = range(num_periods)
        return agent

    def _add_lagged_choice(agent):
        """This function iterates through an agent record and constructs the state
        variables for each point in time."""
        agent["lagged_choice"] = np.nan
        lagged_choice = 3
        for _, row in agent.iterrows():
            period = int(row["Period"])
            agent["Lagged_Choice"].iloc[period] = lagged_choice
            lagged_choice = row["Choice"]

        return agent

    column_labels = []
    column_labels += ["Identifier", "Total_Periods", "Choice", "Reward", "Experience_A"]
    column_labels += ["Experience_B", "Years_Schooling", "Lagged_Choice"]

    matrix = np.array(np.genfromtxt("ftest.txt", missing_values="."), ndmin=2)
    df = pd.DataFrame(matrix, columns=column_labels)

    df = df.groupby("Identifier").apply(_add_period)
    df = df.groupby("Identifier").apply(_add_lagged_choice)

    # We need to construct a wage variable that corresponds to the rewards column if an
    # individual is working.
    cond = df["Choice"].isin([1, 2])
    df["Wage"] = np.nan
    df.loc[cond, "Wage"] = df.loc[cond, "Reward"]

    # Write out resulting sample.
    df = df[DATA_LABELS_EST].astype(DATA_FORMATS_EST)
    with open("ftest.respy.dat", "w") as file_:
        df.to_string(file_, index=False, header=True, na_rep=".")


def write_core_parameters(optim_paras):
    """ This function writes out the core parameters."""
    with open("in.txt", "a") as file_:
        # Write out coefficients for the two occupations.
        coeffs_a, coeffs_b = optim_paras["coeffs_a"], optim_paras["coeffs_b"]
        for coeffs in [coeffs_a, coeffs_b]:
            fmt = " {0:10.6f} {1:10.6f} {2:10.6f} {3:10.6f} {4:10.6f} {5:10.6f}\n"
            file_.write(fmt.format(*coeffs))

        # Write out coefficients for education and home rewards as well as the discount
        # factor. The intercept is scaled. This is later undone again in the original
        # FORTRAN code.
        coeffs_edu, coeffs_home = (
            optim_paras["coeffs_edu"],
            optim_paras["coeffs_home"],
        )

        edu_int = coeffs_edu[0] / 1000
        edu_coeffs = [edu_int]
        home = coeffs_home[0] / 1000
        for j in range(2):
            edu_coeffs += [-coeffs_edu[j + 1] / 1000]
        coeffs = edu_coeffs + [home, optim_paras["delta"][0]]
        fmt = " {0:10.6f} {1:10.6f} {2:10.6f} {3:10.6f} {4:10.6f}\n"
        file_.write(fmt.format(*coeffs))


def transform_respy_to_restud_est(
    optim_paras,
    edu_spec,
    num_agents_est,
    num_draws_prob,
    tau,
    num_periods,
    num_draws_emax,
    cov,
):
    """Transform a RESPY initialization file to a RESTUD file."""
    # Ensure restrictions
    assert edu_spec["start"][0] == 10
    assert edu_spec["max"] == 20

    # Write to initialization file
    with open("in.txt", "w") as file_:

        # Write out some basic information about the problem.
        fmt = (
            " {:03d} {:05d} {:06d}. {:05d}. {:05d}. {:02d} {:02d} {:02d} {:} {:05d} "
            "{:03d}.\n"
        )
        args = [
            num_periods,
            num_agents_est,
            num_draws_emax,
            num_draws_prob,
            int(tau),
            1,
            1,
            1,
            " NO",
            13150,
            40,
        ]
        file_.write(fmt.format(*args))

    write_core_parameters(optim_paras)

    # There is scaling of the Cholesky factors going on that needs to be undone here.
    shocks_cholesky = np.linalg.cholesky(cov)
    for i in [2, 3]:
        for j in range(4):
            if j > i:
                continue
            shocks_cholesky[i, j] = shocks_cholesky[i, j] / 1000.0
    cov = np.matmul(shocks_cholesky, shocks_cholesky.T)

    write_covariance_parameters(cov)


def write_covariance_parameters(cov):
    """THis function writes out the information contained in the covariance matrix."""
    with open("in.txt", "a") as file_:
        # We need to undo the scaling procedure in the RESTUD codes and then write out
        # the correlation matrix and the standard deviations.
        sd = np.sqrt(np.diag(cov)).copy().tolist()
        corr = np.identity(4)

        is_deterministic = np.all(sd) == 0

        if is_deterministic:
            corr[:, :] = 0
        else:
            for i in range(4):
                for j in range(4):
                    if j >= i:
                        continue
                    corr[i, j] = cov[i, j] / (np.sqrt(cov[i, i]) * np.sqrt(cov[j, j]))

        for j in range(4):
            fmt = " {0:10.5f} {1:10.5f} {2:10.5f} {3:10.5f}\n"
            file_.write(fmt.format(*corr[j, :]))
        file_.write(fmt.format(*sd))


def transform_respy_to_restud_sim(
    optim_paras, edu_spec, num_agents_sim, num_periods, num_draws_emax, cov
):
    """Transform a RESPY initialization file to a RESTUD file."""
    # Ensure restrictions
    assert edu_spec["start"][0] == 10
    assert edu_spec["max"] == 20

    # Write to initialization file
    with open("in.txt", "w") as file_:

        # Write out some basic information about the problem.
        fmt = " {0:03d} {1:05d} {2:06d} {3:06f} {4:06f}\n"
        args = [num_periods, num_agents_sim, num_draws_emax, -99.0, 500.0]
        file_.write(fmt.format(*args))

    # Write out coefficients for the two occupations.
    write_core_parameters(optim_paras)

    write_covariance_parameters(cov)


def generate_constraints_dict():
    """Generate the arguments for generate_random_model."""
    point_constr = {
        "program": {"version": "fortran"},
        "estimation": {"maxfun": 0},
        "num_periods": int(np.random.choice(range(2, 10))),
        "edu_spec": {"start": [10], "max": 20, "share": [1.0]},
    }

    bound_constr = {"max_draws": np.random.randint(10, 100)}

    args = {
        "point_constr": point_constr,
        "bound_constr": bound_constr,
        "num_types": 1,
        "deterministic": True,
    }

    return args


def adjust_model_spec(params_spec, options_spec):
    """ This function adjusts the random initialization dictionary further so we can
    campare RESPY against RESTUD."""
    attr = _create_attribute_dictionary(params_spec, options_spec)
    op = attr["optim_paras"]
    op["coeffs_a"][-9:] = [0.0] * 9
    op["coeffs_b"][-9:] = [0.0] * 9
    op["coeffs_edu"][-1] = op["coeffs_edu"][-2]

    op["coeffs_edu"][2] = 0.0
    op["coeffs_edu"][3] = op["coeffs_edu"][4]
    op["coeffs_edu"][5:] = [0.0] * 2
    op["coeffs_common"] = np.array([0.0, 0.0])
    op["coeffs_home"][1:] = [0.0] * 2

    options_spec = _options_spec_from_attributes(attr)
    params_spec = _params_spec_from_attributes(attr)

    return params_spec, options_spec


@pytest.mark.skipif(not IS_FORTRAN, reason="No FORTRAN available")
class TestClass(object):
    """ This class groups together some tests."""

    def test_1(self):
        """Compare simulation results from the RESTUD program and the RESPY package."""
        args = generate_constraints_dict()
        params_spec, options_spec = generate_random_model(**args)
        params_spec, options_spec = adjust_model_spec(params_spec, options_spec)

        # Indicate RESTUD code the special case of zero disturbance.
        open(".restud.testing.scratch", "a").close()

        # We need to indicate to the RESFORT code to rescale the experience covariates.
        open(".restud.respy.scratch", "a").close()

        # Perform toolbox actions
        respy_obj = RespyCls(params_spec, options_spec)

        # This flag aligns the random components between the RESTUD program and RESPY
        # package. The existence of the file leads to the RESTUD program to write out
        # the random components.
        (
            optim_paras,
            edu_spec,
            num_agents_sim,
            num_periods,
            num_draws_emax,
        ) = dist_class_attributes(
            respy_obj,
            "optim_paras",
            "edu_spec",
            "num_agents_sim",
            "num_periods",
            "num_draws_emax",
        )

        shocks_cholesky = optim_paras["shocks_cholesky"]
        cov = np.matmul(shocks_cholesky, shocks_cholesky.T)

        # Simulate sample model using RESTUD code.
        transform_respy_to_restud_sim(
            optim_paras, edu_spec, num_agents_sim, num_periods, num_draws_emax, cov
        )

        # Solve model using RESTUD code.
        cmd = str(TEST_RESOURCES_BUILD / "kw_dp3asim")
        subprocess.check_call(cmd, shell=True)

        # We need to ensure for RESPY that the lagged activity variable indicates that
        # the individuals were in school the period before entering the model.
        types = np.random.choice([3], size=num_agents_sim)
        np.savetxt(".initial_lagged.respy.test", types, fmt="%i")

        # Solve model using RESPY package.
        simulate_observed(respy_obj, is_missings=False)

        # Compare the simulated dataset generated by the programs.
        column_labels = []
        column_labels += ["Experience_A", "Experience_B"]
        column_labels += ["Years_Schooling", "Lagged_Choice"]

        py = pd.read_csv(
            "data.respy.dat",
            delim_whitespace=True,
            header=0,
            na_values=".",
            usecols=column_labels,
        ).astype(np.float)

        fort = pd.DataFrame(
            np.array(np.genfromtxt("ftest.txt", missing_values="."), ndmin=2)[:, -4:],
            columns=column_labels,
        ).astype(np.float)

        # The simulated dataset from FORTRAN includes an indicator for the lagged
        # activities.
        py["Lagged_Choice"] = py["Lagged_Choice"].map({1: 0.0, 2: 0.0, 3: 1.0, 4: 0.0})

        assert_frame_equal(py, fort)

    def test_2(self):
        """Compare results from an evaluation of the criterion function at the initial
        values."""
        args = generate_constraints_dict()
        params_spec, options_spec = generate_random_model(**args)
        params_spec, options_spec = adjust_model_spec(params_spec, options_spec)

        max_draws = args["bound_constr"]["max_draws"]

        # At this point, the random initialization file does only provide diagonal
        # covariances.
        cov_sampled = np.random.uniform(0, 0.01, size=(4, 4)) + np.diag(
            np.random.uniform(1.0, 1.5, size=4)
        )
        chol = np.linalg.cholesky(cov_sampled)
        coeffs = chol[np.tril_indices(4)]
        params_spec.loc["shocks", "para"] = coeffs
        params_spec.loc["shocks", "upper"] = np.nan
        params_spec.loc["shocks", "lower"] = np.nan

        respy_obj = RespyCls(params_spec, options_spec)

        # This flag aligns the random components between the RESTUD program and RESPY
        # package. The existence of the file leads to the RESTUD program to write out
        # the random components.
        (
            optim_paras,
            edu_spec,
            num_agents_est,
            num_periods,
            num_draws_emax,
            num_draws_prob,
            tau,
            num_agents_sim,
        ) = dist_class_attributes(
            respy_obj,
            "optim_paras",
            "edu_spec",
            "num_agents_est",
            "num_periods",
            "num_draws_emax",
            "num_draws_prob",
            "tau",
            "num_agents_sim",
        )

        shocks_cholesky = optim_paras["shocks_cholesky"]
        cov = np.matmul(shocks_cholesky, shocks_cholesky.T)

        # Simulate sample model using RESTUD code.
        transform_respy_to_restud_sim(
            optim_paras, edu_spec, num_agents_sim, num_periods, num_draws_emax, cov
        )

        open(".restud.testing.scratch", "a").close()
        cmd = str(TEST_RESOURCES_BUILD / "kw_dp3asim")
        subprocess.check_call(cmd, shell=True)

        transform_respy_to_restud_est(
            optim_paras,
            edu_spec,
            num_agents_est,
            num_draws_prob,
            tau,
            num_periods,
            num_draws_emax,
            cov,
        )

        filenames = ["in.txt", TEST_RESOURCES_DIR / "in_bottom.txt"]
        with open("in1.txt", "w") as outfile:
            for fname in filenames:
                with open(fname) as infile:
                    outfile.write(infile.read())

        draws_standard = np.random.multivariate_normal(
            np.zeros(4), np.identity(4), (num_periods, max_draws)
        )

        with open(".draws.respy.test", "w") as file_:
            for period in range(num_periods):
                for i in range(max_draws):
                    fmt = " {0:15.10f} {1:15.10f} {2:15.10f} {3:15.10f}\n"
                    line = fmt.format(*draws_standard[period, i, :])
                    file_.write(line)

        # We always need the seed.txt
        shutil.copy(str(TEST_RESOURCES_DIR / "seed.txt"), "seed.txt")
        cmd = str(TEST_RESOURCES_BUILD / "kw_dpml4a")
        subprocess.check_call(cmd, shell=True)
        Path("seed.txt").unlink()

        with open("output1.txt", "r") as searchfile:
            # Search file for strings, trim lines and save as variables
            for line in searchfile:
                if "OLD LOGLF=" in line:
                    stat = float(shlex.split(line)[2])
                    break

        # Now we also evaluate the criterion function with the RESPY package.
        restud_sample_to_respy()
        respy_obj = respy.RespyCls(params_spec, options_spec)
        respy_obj.attr["file_est"] = "ftest.respy.dat"

        open(".restud.respy.scratch", "a").close()
        _, val = respy_obj.fit()
        Path(".restud.respy.scratch").unlink()

        # This ensure that the two values are within 1% of the RESPY value.
        np.testing.assert_allclose(
            abs(stat), abs(val * num_agents_est), rtol=0.01, atol=0.00
        )
