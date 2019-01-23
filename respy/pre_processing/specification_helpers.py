import pandas as pd
from os.path import join, dirname
import numpy as np


def csv_template(num_types, save_path=None, initialize_coeffs=True):
    template = _base_template()
    if num_types > 1:
        to_concat = [
            template,
            _type_share_template(num_types),
            _type_shift_template(num_types),
        ]
        template = pd.concat(to_concat, axis=0, sort=False)
    if initialize_coeffs is False:
        template["para"] = np.nan
    if save_path is not None:
        template.to_csv(save_path)
    return template


def _base_template():
    path = join(dirname(__file__), "base_spec.csv")
    base_template = pd.read_csv(path)
    base_template.set_index(["category", "name"], inplace=True)
    return base_template


def _type_share_template(num_types):
    assert num_types > 1
    types = list(range(1, num_types + 1))
    to_concat = []
    for typ in types[1:]:
        # add the type share coefficients
        ind = ("type_shares", "base_share_{}".format(typ))
        comment = "share_of_agents_of_type_{}".format(typ)
        dat = [1 / num_types, False, np.nan, np.nan, comment]
        to_concat.append(_base_row(index_tuple=ind, data=dat))
        ind = ("type_shares", "ten_years_{}".format(typ))
        comment = (
            "effect of more than ten years of schooling on "
            "probability of being type {}".format(typ)
        )
        dat = [0, False, np.nan, np.nan, comment]
        to_concat.append(_base_row(index_tuple=ind, data=dat))

    return pd.concat(to_concat, axis=0, sort=False)


def _type_shift_template(num_types):
    assert num_types > 1
    types = list(range(1, num_types + 1))
    to_concat = []
    for typ in types[1:]:
        for sector in ["occ_a", "occ_b", "edu", "home"]:
            ind = ("type_shift", "type_{}_in_{}".format(typ, sector))
            comment = "deviation for type {} from type 1 in {}".format(typ, sector)
            dat = [0, False, np.nan, np.nan, comment]
            to_concat.append(_base_row(index_tuple=ind, data=dat))
    return pd.concat(to_concat, axis=0, sort=False)


def _base_row(index_tuple, data):
    ind = pd.MultiIndex.from_tuples([index_tuple], names=["category", "name"])
    cols = ["para", "fixed", "lower", "upper", "comment"]
    df = pd.DataFrame(index=ind, columns=cols, data=[data])
    return df
