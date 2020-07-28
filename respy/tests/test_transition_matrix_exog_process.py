import numpy as np
import pandas as pd
import pytest

from respy.pre_processing.transition_matrix_exog_process import check_numerics


@pytest.fixture(scope="module")
def random_matrix():
    n_states = np.random.randint(1, 100)
    n_process_states = np.random.randint(1, 100)
    df = pd.DataFrame(np.random.randint(0, 100, size=(n_states, n_process_states)))
    df = df.div(np.sum(df, axis=1), axis=0)
    return df


def test_checks(random_matrix):
    check_numerics(random_matrix, random_matrix.shape[0])
