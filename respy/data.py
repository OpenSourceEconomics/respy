import numpy as np
import pandas as pd

from respy.config import TEST_RESOURCES_DIR
import respy.shared as rp_shared


def _create_working_experience(df, options):
    for choice in options["choices_w_wage"]:
        df[f"Experience_{choice.title()}"] = df.Choice.eq(choice)
        df[f"Experience_{choice.title()}"] = (
            df.groupby("Identifier")[f"Experience_{choice.title()}"]
            .cumsum()
            .astype(np.uint8)
        )

    return df


def create_kw_97():
    """Create data for Keane and Wolpin (1997).

    The data includes individuals labor market history and accumulated experiences in
    white-collar, blue-collar occupations, military and schooling.

    """
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

    options = {"choices_w_wage": ["a", "b", "mil"]}
    df = _create_working_experience(df, options)

    df["Lagged_Choice"] = df.groupby("Identifier").Choice.shift(1)

    options["choices_w_exp"] = ["a", "b", "mil", "edu"]
    labels, _ = rp_shared.generate_column_labels_estimation(options)

    df = df.assign(Period=df.Age - 16).drop(columns="Age").loc[:, labels]

    return df


def create_kw_94(df):
    """Convert data to be used with Keane and Wolpin (1994).

    The model in Keane and Wolpin (1994) does not include military as an occupational
    choice. Thus, the history of individuals is truncated the first time they enter the
    military.

    """
    df = create_kw_97()
    # Create indicator if an individual entered the military and from thereon.
    df["Entered_Military"] = df.Choice.eq("mil")
    df.Entered_Military = df.groupby("Identifier").Entered_Military.cumsum()

    df = df.loc[df.Entered_Military.eq(0)].copy()
    df = df.drop(columns=["Entered_Military", "Experience_Mil"])

    # Remove military as choice.
    df.Choice = df.Choice.cat.remove_unused_categories()
    df.Lagged_Choice = df.Lagged_Choice.cat.remove_unused_categories()

    options = {"choices_w_exp": ["a", "b", "edu"]}
    labels, _ = rp_shared.generate_column_labels_estimation(options)

    df = df.reset_index(drop=True).loc[:, labels]

    return df
