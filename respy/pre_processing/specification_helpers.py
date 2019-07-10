import numpy as np
import pandas as pd

from respy.config import ROOT_DIR


def csv_template(n_types=1, save_path=None, initialize_coeffs=True):
    """Creates a template for the parameter specification.

    Parameters
    ----------
    n_types : int, optional
        Number of types in the model. Default is one.
    save_path : str, pathlib.Path, optional
        The template is saved to this path. Default is ``None``.
    initialize_coeffs : bool, optional
        Whether coefficients are initialized with values or not. Default is ``True``.

    """
    template = _base_template()
    if n_types > 1:
        to_concat = [
            template,
            _type_prob_template(n_types),
            _type_shift_template(n_types),
        ]
        template = pd.concat(to_concat, axis=0, sort=False)
    if initialize_coeffs is False:
        template["para"] = np.nan
    if save_path is not None:
        template.to_csv(save_path)
    return template


def _base_template():
    base_template = pd.read_csv(ROOT_DIR / "pre_processing" / "base_params.csv")
    base_template.set_index(["category", "name"], inplace=True)
    return base_template


def _type_prob_template(n_types):
    to_concat = []
    for typ in range(2, n_types + 1):
        # add the type share coefficients
        ind = (f"type_{typ}", "up_to_nine_years_edu")
        comment = (
            "effect of up to nine years of schooling on probability of being "
            f"type {typ}"
        )
        dat = [1 / n_types, False, np.nan, np.nan, comment]
        to_concat.append(_base_row(index_tuple=ind, data=dat))
        ind = (f"type_{typ}", "at_least_ten_years_edu")
        comment = (
            "effect of at least ten years of schooling on probability of being "
            f"type {typ}"
        )
        dat = [0, False, np.nan, np.nan, comment]
        to_concat.append(_base_row(index_tuple=ind, data=dat))

    return pd.concat(to_concat, axis=0, sort=False)


def _type_shift_template(n_types):
    to_concat = []
    for typ in range(2, n_types + 1):
        for choice in ["a", "b", "edu", "home"]:
            ind = ("type_shift", "type_{}_in_{}".format(typ, choice))
            comment = "deviation for type {} from type 1 in {}".format(typ, choice)
            dat = [0, False, np.nan, np.nan, comment]
            to_concat.append(_base_row(index_tuple=ind, data=dat))
    return pd.concat(to_concat, axis=0, sort=False)


def _base_row(index_tuple, data):
    ind = pd.MultiIndex.from_tuples([index_tuple], names=["category", "name"])
    cols = ["para", "fixed", "lower", "upper", "comment"]
    df = pd.DataFrame(index=ind, columns=cols, data=[data])
    return df
