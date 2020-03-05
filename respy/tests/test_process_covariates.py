import io
from textwrap import dedent

import pandas as pd

from respy.pre_processing.process_covariates import remove_irrelevant_covariates


def test_identify_relevant_covariates():
    params = pd.read_csv(
        io.StringIO(
            dedent(
                """
                category,name,value
                wage_a,constant,1
                nonpec_b,upper_upper,1
                wage_c,upper_upper_with_spacing_problem,1
                """
            )
        ),
        index_col=["category", "name"],
    )

    options = {
        "covariates": {
            "constant": "1",
            "nested_covariate": "2",
            "upper": "nested_covariate > 2",
            "upper_upper": "upper == 5",
            "unrelated_covariate": "2",
            "unrelated_covariate_upper": "unrelated_covariate",
            "upper_upper_with_spacing_problem": "upper>2",
        }
    }

    expected = options.copy()
    expected["covariates"].pop("unrelated_covariate")
    expected["covariates"].pop("unrelated_covariate_upper")

    relevant_covariates = remove_irrelevant_covariates(options, params)

    assert expected == relevant_covariates
