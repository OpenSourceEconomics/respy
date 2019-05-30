"""Run a few regression tests."""
import numpy as np
import pandas as pd
import pytest

from development.testing.regression.run_regression import check_single
from respy.config import ROOT_DIR
from respy.pre_processing.model_processing import cov_matrix_to_sdcorr_params


@pytest.mark.parametrize("index", range(100))
def test_single_regression(regression_vault, index):
    """Run a single regression test."""
    test = regression_vault[index]

    # ==================================================================================
    # this is only to keep old regression tests:
    # ==================================================================================
    params_spec, options_spec, crit_val = test
    chol_params = params_spec.loc["shocks", "para"].to_numpy()
    chol = np.zeros((4, 4))
    chol[np.tril_indices(4)] = chol_params
    cov = chol.dot(chol.T)
    helper = pd.read_csv(ROOT_DIR / "pre_processing" / "base_spec2.csv")
    helper.set_index(["category", "name"], inplace=True)
    helper = helper[["old_category", "old_name"]]
    helper_2 = params_spec.loc[["type_shares", "type_shift"]]
    helper_2 = helper_2.reset_index().set_index(["category", "name"], drop=False)
    helper_2.rename(
        columns={"category": "old_category", "name": "old_name"}, inplace=True
    )
    helper_2 = helper_2[["old_category", "old_name"]]
    helper = pd.concat([helper, helper_2], axis=0)
    new_spec = pd.merge(
        helper,
        params_spec,
        left_on=["old_category", "old_name"],
        right_index=True,
        how="left",
    )
    new_spec.loc["shocks", "para"] = cov_matrix_to_sdcorr_params(cov)
    params_spec = new_spec
    test = params_spec, options_spec, crit_val
    # ==================================================================================
    check_single(test)
