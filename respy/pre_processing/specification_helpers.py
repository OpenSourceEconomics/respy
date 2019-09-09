import numpy as np
import pandas as pd

from respy.config import ROOT_DIR


def csv_template(n_types, n_type_covariates, initialize_coeffs=True):
    """Creates a template for the parameter specification.

    Parameters
    ----------
    n_types : int, optional
        Number of types in the model. Default is one.
    n_type_covariates : int, optional
        Number of covariates to predict type probabilities. Can be two or three.
    initialize_coeffs : bool, optional
        Whether coefficients are initialized with values or not. Default is ``True``.

    """
    template = _base_template()
    if n_types > 1:
        to_concat = [
            template,
            _type_prob_template(n_types, n_type_covariates),
            _type_shift_template(n_types),
        ]
        template = pd.concat(to_concat, axis=0, sort=False)
    if initialize_coeffs is False:
        template["value"] = np.nan

    return template


def _base_template():
    base_template = pd.read_csv(ROOT_DIR / "pre_processing" / "base_params.csv")
    base_template.set_index(["category", "name"], inplace=True)
    return base_template


def _type_prob_template(n_types, n_type_covariates):
    to_concat = []
    for type_ in range(2, n_types + 1):
        if n_type_covariates == 3:
            ind = (f"type_{type_}", "constant")
            comment = f"constant effect on probability of being type {type_}"
            dat = [0, comment]
            to_concat.append(_base_row(index_tuple=ind, data=dat))
        else:
            pass

        ind = (f"type_{type_}", "up_to_nine_years_edu")
        comment = (
            "effect of up to nine years of schooling on probability of being "
            f"type {type_}"
        )
        dat = [1 / n_types, comment]
        to_concat.append(_base_row(index_tuple=ind, data=dat))

        ind = (f"type_{type_}", "at_least_ten_years_edu")
        comment = (
            "effect of at least ten years of schooling on probability of being "
            f"type {type_}"
        )
        dat = [0, comment]
        to_concat.append(_base_row(index_tuple=ind, data=dat))

    return pd.concat(to_concat, axis=0, sort=False)


def _type_shift_template(n_types):
    to_concat = []
    for type_ in range(2, n_types + 1):
        for choice in ["a", "b", "edu", "home"]:
            ind = ("type_shift", f"type_{type_}_in_{choice}")
            comment = f"deviation for type {type_} from type 1 in {choice}"
            dat = [0, comment]
            to_concat.append(_base_row(index_tuple=ind, data=dat))
    return pd.concat(to_concat, axis=0, sort=False)


def initial_and_max_experience_template(edu_starts, edu_shares, edu_max):
    to_concat = []
    for start, share in zip(edu_starts, edu_shares):
        ind = ("initial_exp_edu", start)
        dat = [share, f"Probability that the initial level of education is {start}."]
        to_concat.append(_base_row(ind, dat))

    ind = ("maximum_exp", "edu")
    dat = [edu_max, "Maximum level of experience for education"]
    to_concat.append(_base_row(ind, dat))

    return pd.concat(to_concat, axis=0, sort=False)


def lagged_choices_template(n_lagged_choices):
    to_concat = []
    for i in range(1, n_lagged_choices + 1):
        ind = (f"lagged_choice_{i}_edu", "up_to_nine_years_of_edu")
        dat = [1, ""]
        to_concat.append(_base_row(ind, dat))
        ind = (f"lagged_choice_{i}_edu", "at_least_ten_years_edu")
        dat = [0, ""]
        to_concat.append(_base_row(ind, dat))

    return pd.concat(to_concat, axis=0, sort=False)


def _base_row(index_tuple, data):
    ind = pd.MultiIndex.from_tuples([index_tuple], names=["category", "name"])
    cols = ["value", "comment"]
    df = pd.DataFrame(index=ind, columns=cols, data=[data])
    return df
