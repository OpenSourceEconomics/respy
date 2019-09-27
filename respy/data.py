import numpy as np
import pandas as pd

from respy import shared as rp_shared
from respy.config import TEST_RESOURCES_DIR
from respy.pre_processing.model_processing import process_params_and_options


def _create_working_experience(df, options):
    for choice in options["choices_w_wage"]:
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
    _, options = process_params_and_options(params, options)

    columns = ["Identifier", "Age", "Experience_Edu", "Choice", "Wage"]
    dtypes = {
        "Identifier": np.int,
        "Age": np.uint8,
        "Experience_Edu": np.uint8,
        "Choice": "category",
    }

    df = pd.read_csv(
        TEST_RESOURCES_DIR / "KW_97.raw",
        sep=r"\s+",
        names=columns,
        dtype=dtypes,
        na_values=".",
        float_precision="high",
    )

    df.Identifier = df.groupby("Identifier").ngroup().astype(np.uint16)

    codes_to_choices = {"3": "a", "4": "b", "5": "mil", "1": "edu", "2": "home"}
    df.Choice = df.Choice.cat.set_categories(codes_to_choices).cat.rename_categories(
        codes_to_choices
    )

    df = _create_working_experience(df, options)

    df["Lagged_Choice_1"] = df.groupby("Identifier").Choice.shift(1)

    labels, _ = rp_shared.generate_column_labels_estimation(options)

    df = df.assign(Period=df.Age - 16).drop(columns="Age").loc[:, labels]

    return df
