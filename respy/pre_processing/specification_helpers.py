import numpy as np
import pandas as pd

from respy.config import ROOT_DIR


def csv_template(n_types, n_type_covariates, observables, initialize_coeffs=True):
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

    if observables is not False:
        to_concat = [
            template,
            _observable_template(observables)
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


def lagged_choices_probs_template(n_lagged_choices, choices):
    to_concat = []
    for i in range(1, n_lagged_choices + 1):
        probs = np.random.uniform(size=len(choices))
        probs /= probs.sum()
        for j, choice in enumerate(choices):
            ind = (f"lagged_choice_{i}_{choice}", "constant")
            dat = [probs[j], f"Probability of choice {choice} being lagged choice {i}"]
            to_concat.append(_base_row(ind, dat))

    return pd.concat(to_concat, axis=0, sort=False)


def lagged_choices_covariates_template(n_lagged_choices, choices):
    return {
        "not_exp_a_lagged": "exp_a > 0 and lagged_choice_1 != 'a'",
        "not_exp_b_lagged": "exp_b > 0 and lagged_choice_1 != 'b'",
        "work_a_lagged": "lagged_choice_1 == 'a'",
        "work_b_lagged": "lagged_choice_1 == 'b'",
        "edu_lagged": "lagged_choice_1 == 'edu'",
        "is_return_not_high_school": "~edu_lagged and ~hs_graduate",
        "is_return_high_school": "~edu_lagged and hs_graduate",
    }


def _base_row(index_tuple, data):
    ind = pd.MultiIndex.from_tuples([index_tuple], names=["category", "name"])
    cols = ["value", "comment"]
    df = pd.DataFrame(index=ind, columns=cols, data=[data])
    return df

def _observable_template(observables):
    to_concat = []
    for x in range(len(observables)):
        probs = np.random.uniform(size=observables[x])
        probs /= probs.sum()
        for y in range(observables[x]):
            ind = (f"observables", "{}observable_{}".format(x,y))
            dat = [probs[y], f"Probability of observable {x} being level choice {y}"]
            to_concat.append(_base_row(ind, dat))
    return pd.concat(to_concat, axis=0, sort=False)