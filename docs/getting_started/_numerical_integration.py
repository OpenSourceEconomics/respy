"""Auxiliary and plotting functions for the respy tutorial numerical integration."""
import matplotlib.pyplot as plt
from matplotlib import ticker

# Initialize methods for comparison of integration
METHODS = ["random", "halton", "sobol"]
TUITION_SUBSIDIES = [0, 500]


def get_single_integrand(
    df, idx, scaling, methods, manual_limits="no", xlimits=[0, 1], ylimits=[0, 1]
):
    """Plots the values of the idx-th evaluation of an integral stored in a
    dataframe. The dataframe contains values of the integrand computed at
    a different number of points.

    Input
        - df (dataframe): dataframe of [integral] values
        - idx (scalar): number of integral evaluation
        - scaling (scalar): scaling of [integral] values (1 = no scaling)
        - manual_limits (string): determine whether to set manual limits
        - methods (list): integration methods
    Output
        - figure
    """

    fig, ax = plt.subplots()

    for m in methods:
        label = get_label(m)
        x = df.index.get_level_values("integration_points").unique()
        y = df.loc[(slice(None), idx), m] / scaling

        ax.plot(x, y, label=label)

    ax.legend()
    if manual_limits == "yes":
        ax.set_xlim(xlimits)
        ax.set_ylim(ylimits)
    ax.set_xlabel("Points")
    ax.set_ylabel("Integrand")


def get_label(method):
    """Standardizes labels for graphs given METHOD list

    Input
        - method (list): list of methods to standardize
    Output
        - label (string): standardizes labels
    """

    label = method.capitalize()
    if label == "Random":
        label = "random"

    return label


def get_rmse_rate(df, comparison_rates, methods):
    """Plots the absolute rate of rmse on a loglog scale for different METHODS
    of function/integral value evaluation.

    Input
        - df (dataframe): calculated [RMSE] values for various number of points
        - comparison_rates (list): list of reference rates

    Output
        - figure
    """

    for measure in ["absolute", "relative"]:

        fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))

        for m in methods:
            label = get_label(m)

            x = df.index.get_level_values("integration_points").unique()
            y = df.loc[measure, m]

            ax.loglog(x, y, label=label)

        ax.loglog(
            x,
            comparison_rates[0],
            label="$C_1 log(n)/n$",
            linestyle="-.",
            linewidth=2,
            color="silver",
        )
        ax.loglog(
            x,
            comparison_rates[1],
            label="$C_2 log(n)/n^{1/2}$",
            linestyle="-.",
            linewidth=2,
            color="grey",
        )
        ax.loglog(
            x,
            comparison_rates[2],
            label="$C_3 log(n)/n^{3/2}$",
            linestyle="-.",
            linewidth=2.5,
            color="darkviolet",
        )

        ax.set_xlabel("Points"), ax.legend()
        ax.set_ylabel(f"Rate {measure} RMSE")


def get_policy_prediction(df, baseline, alternative, methods):
    """Plots the policy prediction under different integrations methods.

    Input
        - df (dataframe): policy-values for various number of points
        - baseline (scalar): location of baseline values in the df
        - alternative (scalar): location of policy values
        - methods (list): integration methods

    Output
        figure
    """

    fig, ax = plt.subplots()

    for m in methods:
        label = get_label(m)

        x = df.index.get_level_values("Points").unique()
        y = df.loc[(m, slice(None)), baseline] - df.loc[(m, slice(None)), alternative]
        ax.plot(x, y, label=label)

    ax.set_xlabel("Points")
    set_formatter(ax, which="x")

    ax.set_ylabel(r"$\Delta$ Schooling ")
    ax.legend()


def set_formatter(ax, is_int=True, which="xy"):
    """Formats axis values.

    Input
        - ax: which ax object to format
        - which: axis

    Output
        - formatted ax object
    """
    formatter = ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    if "x" in which:
        ax.get_xaxis().set_major_formatter(formatter)
    if "y" in which:
        ax.get_yaxis().set_major_formatter(formatter)
