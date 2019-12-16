"""Everything related to the original data from Keane and Wolpin (1997)."""
import numpy as np
import pandas as pd

from respy import shared as rp_shared
from respy.config import TEST_RESOURCES_DIR
from respy.pre_processing.model_processing import process_params_and_options


def _create_working_experience(df, optim_paras):
    for choice in optim_paras["choices_w_wage"]:
        df[f"Experience_{choice.title()}"] = df.Choice.eq(choice)
        df[f"Experience_{choice.title()}"] = (
            df.groupby("Identifier")[f"Experience_{choice.title()}"]
            .shift()
            .fillna(0)
            .astype(np.uint8)
        )
        df[f"Experience_{choice.title()}"] = df.groupby("Identifier")[
            f"Experience_{choice.title()}"
        ].cumsum()

    return df


def create_kw_97(params, options):
    """Create data for Keane and Wolpin (1997).

    The data includes individuals labor market history and accumulated experiences in
    white-collar, blue-collar occupations, military and schooling.

    """
    optim_paras, options = process_params_and_options(params, options)

    dtypes = {
        "Identifier": np.int,
        "Age": np.uint8,
        "Experience_School": np.uint8,
        "Choice": "category",
        "Wage": np.float,
    }

    df = pd.read_csv(
        TEST_RESOURCES_DIR / "kw_97_data.csv", dtype=dtypes, float_precision="high"
    )

    df.Identifier = df.groupby("Identifier").ngroup().astype(np.uint16)

    codes_to_choices = {
        "3": "white_collar",
        "4": "blue_collar",
        "5": "military",
        "1": "school",
        "2": "home",
    }
    df.Choice = df.Choice.cat.set_categories(codes_to_choices).cat.rename_categories(
        codes_to_choices
    )

    df = _create_working_experience(df, optim_paras)

    df["Lagged_Choice_1"] = df.groupby("Identifier").Choice.shift(1)
    df["Period"] = df.Age - 16
    df = df.query("Age >= 16")

    labels, _ = rp_shared.generate_column_labels_estimation(optim_paras)

    df = df[labels]

    return df
