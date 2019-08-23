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
    _, _, options = process_params_and_options(params, options)

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

    df["Lagged_Choice"] = df.groupby("Identifier").Choice.shift(1)

    labels, _ = rp_shared.generate_column_labels_estimation(options)

    df = df.assign(Period=df.Age - 16).drop(columns="Age").loc[:, labels]

    return df


def create_kw_94(params, options):
    """Convert data to be used with Keane and Wolpin (1994).

    The model in Keane and Wolpin (1994) does not include military as an occupational
    choice. Thus, the history of individuals is truncated the first time they enter the
    military.

    """
    _, _, options = process_params_and_options(params, options)

    df = create_kw_97(options)
    # Create indicator if an individual entered the military and from thereon.
    df["Entered_Military"] = df.Choice.eq("mil")
    df.Entered_Military = df.groupby("Identifier").Entered_Military.cumsum()

    df = df.loc[df.Entered_Military.eq(0)].copy()
    df = df.drop(columns=["Entered_Military", "Experience_Mil"])

    # Remove military as choice.
    df.Choice = df.Choice.cat.remove_unused_categories()
    df.Lagged_Choice = df.Lagged_Choice.cat.remove_unused_categories()

    labels, _ = rp_shared.generate_column_labels_estimation(options)

    df = df.reset_index(drop=True).loc[:, labels]

    # reset the Identifier such that it starts at 0 and always increments by 1
    df["Identifier"] = pd.Categorical(df["Identifier"]).codes

    return df
