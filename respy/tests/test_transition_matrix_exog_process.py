import numpy as np
import pandas as pd
import pytest

from respy.pre_processing.transition_matrix_exog_process import check_numerics
from respy.pre_processing.transition_matrix_exog_process import create_covariates
from respy.pre_processing.transition_matrix_exog_process import transform_matrix

PROCESS_NAME = "health_shocks"
PROCESS_STATES = ["sick", "healthy"]


@pytest.fixture(scope="module")
def random_matrix():
    n_states = np.random.randint(2, 100)
    n_process_states = np.random.randint(2, 100)
    df = pd.DataFrame(np.random.randint(0, 100, size=(n_states, n_process_states)))
    return df


@pytest.fixture(scope="module")
def states_in():
    states = {}
    states["dependent_process"] = [
        "sick_and_young",
        "sick_and_old",
        "healthy_and_young",
        "healthy_and_old",
    ]
    states["independent_process"] = ["young", "old"]
    states["false_process"] = ["sick_healthy", "sick"]
    return states


@pytest.fixture(scope="module")
def covariates_out():
    covariates = {}
    covariates["dependent_process"] = {
        "sick_and_young": "health_shocks == sick & ?",
        "sick_and_old": "health_shocks == sick & ?",
        "healthy_and_young": "health_shocks == healthy & ?",
        "healthy_and_old": "health_shocks == healthy & ?",
    }
    covariates["independent_process"] = {"young": "?", "old": "?"}
    return covariates


def test_checks(random_matrix):
    random_matrix = random_matrix.div(np.sum(random_matrix, axis=1), axis=0)
    check_numerics(random_matrix, random_matrix.shape[0])


def test_fails_checks(random_matrix):
    with pytest.raises(ValueError):
        check_numerics(random_matrix, random_matrix.shape[0])


@pytest.mark.parametrize("process_type", ["dependent_process", "independent_process"])
def test_covariates_creation_dependent_process(states_in, covariates_out, process_type):
    covs = create_covariates(states_in[process_type], PROCESS_NAME, PROCESS_STATES)
    assert covs == covariates_out[process_type]


def test_fail_creation(states_in):
    with pytest.raises(ValueError):
        create_covariates(states_in["false_process"], PROCESS_NAME, PROCESS_STATES)


def test_transform_matrix(random_matrix):
    matrix = random_matrix.div(np.sum(random_matrix, axis=1), axis=0)
    log_matrix = transform_matrix(matrix)
    check_numerics(np.exp(log_matrix), matrix.shape[0])
