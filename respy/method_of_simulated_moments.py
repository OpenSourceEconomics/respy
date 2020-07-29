"""Estimate models with the method of simulated moments (MSM).

The method of simulated moments is developed by [1]_, [2]_, and [3]_ and an estimation
technique where the distance between the moments of the actual data and the moments
implied by the model parameters is minimized.

References
----------
.. [1] McFadden, D. (1989). A method of simulated moments for estimation of discrete
       response models without numerical integration. Econometrica: Journal of the
       Econometric Society, 995-1026.
.. [2] Lee, B. S., & Ingram, B. F. (1991). Simulation estimation of time-series models.
       Journal of Econometrics, 47(2-3), 197-205.
.. [3] Duffie, D., & Singleton, K. (1993). Simulated Moments Estimation of Markov Models
       of Asset Prices. Econometrica, 61(4), 929-952.

"""
import copy
import functools

import numpy as np
import pandas as pd

from respy.simulate import get_simulate_func


def get_moment_errors_func(
    params,
    options,
    calc_moments,
    replace_nans,
    empirical_moments,
    weighting_matrix,
    n_simulation_periods=None,
    return_scalar=True,
    return_simulated_moments=False,
    return_comparison_plot_data=False,
):
    """Get the moment errors function for MSM estimation.

    Parameters
    ----------
    params : pandas.DataFrame or pandas.Series
        Contains parameters.
    options : dict
        Dictionary containing model options.
    calc_moments : callable or list or dict
        Function(s) used to calculate simulated moments. Must match structure
        of empirical moments i.e. if empirical_moments is a list of
        pandas.DataFrames, calc_moments must be a list of the same length
        containing functions that correspond to the moments in
        empirical_moments.
    replace_nans : callable or list or dict or None
        Functions(s) specifying how to handle missings in simulated_moments.
        Must match structure of empirical_moments.
        Exception: If only one replacement function is specified, it will be
        used on all sets of simulated moments.
    empirical_moments : pandas.DataFrame or pandas.Series or dict or list
        Contains the empirical moments calculated for the observed data. Moments
        should be saved to pandas.DataFrame or pandas.Series that can either be
        passed to the function directly or as items of a list or dictionary.
        Index of pandas.DataFrames can be of type MultiIndex, but columns cannot.
    weighting_matrix : numpy.ndarray
        Square matrix of dimension (NxN) with N denoting the number of
        empirical_moments. Used to weight squared moment errors.
    n_simulation_periods : int, default None
        Dictates the number of periods in the simulated dataset.
        This option does not affect ``options["n_periods"]`` which controls the
        number of periods for which decision rules are computed.
    return_scalar : bool, default True
        Indicates whether to return moment error vector (False) or weighted
        square product of moment error vectors (True).
    return_simulated_moments: bool, default False
        Indicates whether simulated moments should be returned with other output.
        If True will return simulated moments of the same type as empirical_moments.
    return_comparison_plot_data: bool, default False
        Indicator for whether a :class:`pandas.DataFrame` with empirical and simulated
        moments for the visualization with estimagic should be returned. Data contains
        the following columns:
        - moment_column: Contains the column names of the moment DataFrames/Series
        names.
        - moment_index: Contains the index of the moment DataFrames/Series. MultiIndex
        indices will be joined to one string.
        - value: Contains moment values.
        - moment_set: Indicator for each set of moments, will use keys if
        empirical_moments are specified in a dict. Moments input as lists will
        be numbered according to position.
        - kind: Indicates whether moments are empirical or simulated.

    Returns
    -------
    moment_errors_func: callable()
         Function where all arguments except the parameter vector are set.

    """
    empirical_moments = copy.deepcopy(empirical_moments)
    input_type = type(empirical_moments)

    simulate = get_simulate_func(
        params=params, options=options, n_simulation_periods=n_simulation_periods
    )

    empirical_moments = _harmonize_input(empirical_moments)
    calc_moments = _harmonize_input(calc_moments)

    # If only one replacement function is given for multiple sets of moments,
    # duplicate replacement function for all sets of simulated moments.
    if replace_nans is not None:
        if callable(replace_nans):
            replace_nans = {k: replace_nans for k in empirical_moments}
        replace_nans = _harmonize_input(replace_nans)

        if 1 < len(replace_nans) < len(empirical_moments):
            raise ValueError(
                "Replacement functions can only be matched 1:1 or 1:n with sets of "
                "empirical moments."
            )

        elif len(replace_nans) > len(empirical_moments):
            raise ValueError(
                "There are more replacement functions than sets of empirical moments."
            )

        else:
            pass

    if len(calc_moments) != len(empirical_moments):
        raise ValueError(
            "Number of functions to calculate simulated moments must be equal to "
            "the number of sets of empirical moments."
        )

    if return_simulated_moments and return_comparison_plot_data:
        raise ValueError(
            "Can only return either simulated moments or comparison plot data, not both."
        )

    moment_errors_func = functools.partial(
        moment_errors,
        simulate=simulate,
        calc_moments=calc_moments,
        replace_nans=replace_nans,
        empirical_moments=empirical_moments,
        weighting_matrix=weighting_matrix,
        return_scalar=return_scalar,
        return_simulated_moments=[return_simulated_moments, input_type],
        return_comparison_plot_data=return_comparison_plot_data,
    )

    return moment_errors_func


def moment_errors(
    params,
    simulate,
    calc_moments,
    replace_nans,
    empirical_moments,
    weighting_matrix,
    return_scalar,
    return_simulated_moments,
    return_comparison_plot_data,
):
    """Loss function for MSM estimation.

    Parameters
    ----------
    params : pandas.DataFrame or pandas.Series
        Contains model parameters.
    simulate : callable
        Function used to simulate data for MSM estimation.
    calc_moments : dict
        Dictionary of function(s) used to calculate simulated moments. Must match
        length of empirical_moments i.e. calc_moments contains a moments function for
        each item in empirical_moments.
    replace_nans : dict or None
        Dictionary of functions(s) specifying how to handle missings in
        simulated_moments. Must match length of empirical_moments.
    empirical_moments : dict
        Contains the empirical moments calculated for the observed data. Each item in
        the dict constitutes a set of moments saved to a pandas.DataFrame or
        pandas.Series. Index of pandas.DataFrames can be of type MultiIndex, but columns
        cannot.
    weighting_matrix : numpy.ndarray
        Square matrix of dimension (NxN) with N denoting the number of
        empirical_moments. Used to weight squared moment errors.
    return_scalar : bool
        Indicates whether to return moment error vector (False) or weighted square
        product of moment error vector (True).
    return_simulated_moments: list
        Indicates whether simulated moments should be returned with other output.
        If the first element in the list is True will return simulated moments of
        the same type as specified in the second list element.
    return_comparison_plot_data: bool
        Will output moments in a tidy data format if True. Uses dictionary keys
        from empirical moments to group sets of moments.

    Returns
    -------
    out : pandas.Series or float or tuple
        Scalar or moment error vector depending on value of return_scalar. Will be a
        tuple containing simulated moments of same type as empirical_moments or a tidy
        pandas.DataFrame if either return_simulated_moments or the first element in
        return_comparison_plot_data is True.

    """
    empirical_moments = copy.deepcopy(empirical_moments)

    df = simulate(params)

    simulated_moments = {name: func(df.copy()) for name, func in calc_moments.items()}

    simulated_moments = {
        name: sim_mom.reindex_like(empirical_moments[name])
        for name, sim_mom in simulated_moments.items()
    }

    if replace_nans is not None:
        simulated_moments = {
            name: replace_nans[name](sim_mom)
            for name, sim_mom in simulated_moments.items()
        }

    flat_empirical_moments = _flatten_index(empirical_moments)
    flat_simulated_moments = _flatten_index(simulated_moments)

    moment_errors = flat_empirical_moments - flat_simulated_moments

    # Return moment errors as indexed DataFrame or calculate weighted square product of
    # moment errors depending on return_scalar.
    if return_scalar:
        out = moment_errors.T @ weighting_matrix @ moment_errors
    else:
        out = pd.Series(
            moment_errors @ np.sqrt(weighting_matrix), index=moment_errors.index
        )

    if return_simulated_moments[0]:
        simulated_moments = _reconstruct_input(
            simulated_moments, return_simulated_moments[1]
        )
        out = (out, simulated_moments)

    elif return_comparison_plot_data:
        tidy_moments = _create_comparison_plot_data_msm(
            empirical_moments, simulated_moments
        )
        out = (out, tidy_moments)

    else:
        pass

    return out


def get_diag_weighting_matrix(empirical_moments, weights=None):
    """Create a diagonal weighting matrix from weights.

    Parameters
    ----------
    empirical_moments : pandas.DataFrame or pandas.Series or dict or list
        Contains the empirical moments calculated for the observed data. Moments should
        be saved to pandas.DataFrame or pandas.Series that can either be passed to the
        function directly or as items of a list or dictionary.
    weights : pandas.DataFrame or pandas.Series or dict or list
        Contains weights (usually variances) of empirical moments. Must match structure
        of empirical_moments i.e. if empirical_moments is a list of pandas.DataFrames,
        weights must be list of pandas.DataFrames as well where each DataFrame entry
        contains the weight for the corresponding moment in empirical_moments.

    Returns
    -------
    numpy.ndarray
        Array contains a diagonal weighting matrix.

    """
    weights = copy.deepcopy(weights)
    empirical_moments = copy.deepcopy(empirical_moments)
    empirical_moments = _harmonize_input(empirical_moments)

    # Use identity matrix if no weights are specified.
    if weights is None:
        flat_weights = _flatten_index(empirical_moments)
        flat_weights[:] = 1

    # Harmonize input weights.
    else:
        weights = _harmonize_input(weights)

        # Reindex weights to ensure they are assigned to the correct moments in
        # the MSM function.
        weights = {
            name: weight.reindex_like(empirical_moments[name])
            for name, weight in weights.items()
        }

        flat_weights = _flatten_index(weights)

    return np.diag(flat_weights)


def get_flat_moments(empirical_moments):
    """Compute the empirical moments flat indexes.

    Parameters
    ----------
    empirical_moments : pandas.DataFrame or pandas.Series or dict or list
        containing pandas.DataFrame or pandas.Series. Contains the empirical moments
        calculated for the observed data. Moments should be saved to pandas.DataFrame or
        pandas.Series that can either be passed to the function directly or as items of
        a list or dictionary.

    Returns
    -------
    flat_empirical_moments : pandas.DataFrame
        Vector of empirical_moments with flat index.

    """
    empirical_moments = copy.deepcopy(empirical_moments)
    empirical_moments = _harmonize_input(empirical_moments)
    flat_empirical_moments = _flatten_index(empirical_moments)

    return flat_empirical_moments


def _harmonize_input(data):
    """Harmonize different types of inputs by turning all inputs into dicts.

    - pandas.DataFrames/Series and callable functions will turn into a dict containing a
      single item (i.e. the input).
    - Dictionaries will be left as is.

    """
    # Convert single pandas.DataFrames, pandas.Series or function into dict containing
    # one item.
    if isinstance(data, (pd.DataFrame, pd.Series)) or callable(data):
        data = {0: data}

    # Turn lists into dictionary.
    elif isinstance(data, list):
        data = {i: data_ for i, data_ in enumerate(data)}

    elif isinstance(data, dict):
        pass

    else:
        raise TypeError(
            "Function only accepts lists, dictionaries, functions, Series and "
            "DataFrames as inputs."
        )

    return data


def _flatten_index(data):
    """Flatten the index as a combination of the former index and the columns."""
    data = copy.deepcopy(data)
    data_flat = []

    for name, series_or_df in data.items():
        series_or_df.index = series_or_df.index.map(str)
        # Unstack pandas.DataFrames and pandas.Series to add
        # columns/name to index.
        if isinstance(series_or_df, pd.DataFrame):
            df = series_or_df.rename(columns=lambda x: f"{name}_{x}")
        # pandas.Series without a name are named using a counter to avoid duplicate
        # indexes.
        elif isinstance(series_or_df, pd.Series):
            df = series_or_df.to_frame(name=f"{name}")
        else:
            raise NotImplementedError

        # Columns to the index.
        df = df.unstack()
        df.index = df.index.to_flat_index().str.join("_")
        data_flat.append(df)

    return pd.concat(data_flat)


def _create_comparison_plot_data_msm(empirical_moments, simulated_moments):
    """Create pandas.DataFrame for estimagic's comparison plot."""
    tidy_empirical_moments = _create_tidy_data(empirical_moments)
    tidy_simulated_moments = _create_tidy_data(simulated_moments)

    tidy_simulated_moments["kind"] = "simulated"
    tidy_empirical_moments["kind"] = "empirical"

    return pd.concat(
        [tidy_empirical_moments, tidy_simulated_moments], ignore_index=True
    )


def _create_tidy_data(data):
    """Create tidy data from dict of pandas.DataFrames."""
    tidy_data = []
    for name, series_or_df in data.items():
        # Join index levels for MultiIndex objects.
        if isinstance(series_or_df.index, pd.MultiIndex):
            series_or_df = series_or_df.rename(index=str)
            series_or_df.index = series_or_df.index.to_flat_index().str.join("_")
        # If moments are a pandas.Series, convert into pandas.DataFrame.
        if isinstance(series_or_df, pd.Series):
            # Unnamed pandas.Series receive a name based on a counter.
            series_or_df = series_or_df.to_frame(name=f"{name}")

        # Create pandas.DataFrame in tidy format.
        tidy_df = series_or_df.unstack()
        tidy_df.index.names = ("moment_column", "moment_index")
        tidy_df.rename("value", inplace=True)
        tidy_df = tidy_df.reset_index()
        tidy_df["moment_set"] = name
        tidy_data.append(tidy_df)

    return pd.concat(tidy_data, ignore_index=True)


def _reconstruct_input(data, input_type):
    """Reconstruct input from dict back to a list or single object."""
    if input_type == dict:
        output = data
    else:
        output = list(data.values())

        if len(output) == 1:
            output = output[0]

    return output
