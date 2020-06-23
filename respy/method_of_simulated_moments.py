"""Estimate models with the method of simulated moments (MSM).

The method of simulated moments is developed by [1], [2], and [3] and an estimation
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
import itertools

import numpy as np
import pandas as pd

from respy.simulate import get_simulate_func


def get_msm_func(
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
    """Get the MSM function.

    Parameters
    ----------
    params : pandas.DataFrame or pandas.Series
        Contains parameters.
    options : dict
        Dictionary containing model options.
    calc_moments : callable or list
        Function(s) used to calculate simulated moments. Must match structure
        of empirical moments i.e. if empirical_moments is a list of
        pandas.DataFrames, calc_moments must be a list of the same length
        containing functions that correspond to the moments in
        empirical_moments.
    replace_nans : callable or list
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
    msm_func: callable
        MSM function where all arguments except the parameter vector are set.

    """
    empirical_moments = copy.deepcopy(empirical_moments)

    # Save keys of dictionary for comparison plot if applicable.
    return_comparison_plot_data = [return_comparison_plot_data]
    if isinstance(empirical_moments, dict):
        moment_keys = sorted(empirical_moments)
        return_comparison_plot_data.append(moment_keys)
    else:
        return_comparison_plot_data.append(None)

    simulate = get_simulate_func(
        params=params, options=options, n_simulation_periods=n_simulation_periods
    )

    empirical_moments = _harmonize_input(empirical_moments)
    calc_moments = _harmonize_input(calc_moments)
    replace_nans = _harmonize_input(replace_nans)

    # If only one replacement function is given for multiple sets of moments, duplicate
    # replacement function for all sets of simulated moments.
    if len(replace_nans) == 1 and len(empirical_moments) > 1:
        replace_nans = replace_nans * len(empirical_moments)

    elif 1 < len(replace_nans) < len(empirical_moments):
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

    if return_simulated_moments and return_comparison_plot_data[0]:
        raise ValueError(
            "Can only return either simulated moments or comparison plot data, not both."
        )

    msm_func = functools.partial(
        msm,
        simulate=simulate,
        calc_moments=calc_moments,
        replace_nans=replace_nans,
        empirical_moments=empirical_moments,
        weighting_matrix=weighting_matrix,
        return_scalar=return_scalar,
        return_simulated_moments=return_simulated_moments,
        return_comparison_plot_data=return_comparison_plot_data,
    )

    return msm_func


def msm(
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
    calc_moments : list
        List of function(s) used to calculate simulated moments. Must match length of
        empirical_moments i.e. calc_moments contains a moments function for each item in
        empirical_moments.
    replace_nans : list
        List of functions(s) specifying how to handle missings in simulated_moments.
        Must match length of empirical_moments.
    empirical_moments : list
        Contains the empirical moments calculated for the observed data. Each item in
        the list constitutes a set of moments saved to a pandas.DataFrame or
        pandas.Series. Index of pandas.DataFrames can be of type MultiIndex, but columns
        cannot.
    weighting_matrix : numpy.ndarray
        Square matrix of dimension (NxN) with N denoting the number of
        empirical_moments. Used to weight squared moment errors.
    return_scalar : bool
        Indicates whether to return moment error vector (False) or weighted square
        product of moment error vector (True).
    return_simulated_moments: bool
        Indicates whether simulated moments should be returned with other output.
        If True will return simulated moments of the same type as empirical_moments.
    return_comparison_plot_data: list
        Will output moments in a tidy data format if True. Expects a list as input where
        the first element is a boolean indicating whether to return the comparison plot
        data. The second element in the list can be a list of keys used to identify sets
        of moments which otherwise will be numbered.

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

    simulated_moments = [func(df) for func in calc_moments]

    simulated_moments = [
        sim_mom.reindex_like(emp_mom)
        for emp_mom, sim_mom in zip(empirical_moments, simulated_moments)
    ]

    simulated_moments = [
        func(mom) for mom, func in zip(simulated_moments, replace_nans)
    ]

    flat_empirical_moments = _flatten_index(empirical_moments)
    flat_simulated_moments = _flatten_index(simulated_moments)

    moment_errors = flat_empirical_moments - flat_simulated_moments

    # Return moment errors as indexed DataFrame or calculate weighted square product of
    # moment errors depending on return_scalar.
    if return_scalar:
        out = moment_errors.T @ weighting_matrix @ moment_errors
    else:
        out = moment_errors

    if return_simulated_moments:
        simulated_moments = _reconstruct_inputs(
            simulated_moments, return_comparison_plot_data[1]
        )
        out = (out, simulated_moments)

    elif return_comparison_plot_data[0]:
        tidy_moments = _create_comparison_plot_data_msm(
            empirical_moments, simulated_moments, return_comparison_plot_data[1]
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
        weights be list of pandas.DataFrames as well where each DataFrame entry contains
        the weight for the corresponding moment in empirical_moments.

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
        weights = [
            weight.reindex_like(emp_mom)
            for emp_mom, weight in zip(empirical_moments, weights)
        ]

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
    """Harmonize different types of inputs by turning all inputs into lists.

    - pandas.DataFrames/Series and callable functions will turn into a list containing a
      single item (i.e. the input).
    - Dictionaries will be sorted according to keys and then turn into a list containing
      the dictionary entries.

    """
    # Convert single pandas.DataFrames, pandas.Series or function into list containing
    # one item.
    if isinstance(data, (pd.DataFrame, pd.Series)) or callable(data):
        data = [data]

    # Sort dictionary according to keys and turn into list.
    elif isinstance(data, dict):
        data = [data[key] for key in sorted(data)]

    elif isinstance(data, list):
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
    counter = itertools.count()

    for series_or_df in data:
        series_or_df.index = series_or_df.index.map(str)
        # Unstack pandas.DataFrames and pandas.Series to add
        # columns/name to index.
        if isinstance(series_or_df, pd.DataFrame):
            df = series_or_df.rename(columns=str)
        # pandas.Series without a name are named using a counter to avoid duplicate
        # indexes.
        elif isinstance(series_or_df, pd.Series) and series_or_df.name is None:
            df = series_or_df.to_frame(name=str(next(counter)))
        else:
            df = series_or_df.to_frame(str(series_or_df.name))

        # Columns to the index.
        df = df.unstack()
        df.index = df.index.to_flat_index().str.join("_")
        data_flat.append(df)

    return pd.concat(data_flat)


def _create_comparison_plot_data_msm(
    empirical_moments, simulated_moments, moment_set_labels
):
    """Create pandas.DataFrame for estimagic's comparison plot."""
    if moment_set_labels is None:
        moment_set_labels = list(range(0, len(empirical_moments)))

    tidy_empirical_moments = _create_tidy_data(empirical_moments, moment_set_labels)
    tidy_simulated_moments = _create_tidy_data(simulated_moments, moment_set_labels)

    tidy_simulated_moments["kind"] = "simulated"
    tidy_empirical_moments["kind"] = "empirical"

    return pd.concat(
        [tidy_empirical_moments, tidy_simulated_moments], ignore_index=True
    )


def _create_tidy_data(data, moment_set_labels):
    """Create tidy data from list of pandas.DataFrames."""
    counter = itertools.count()
    tidy_data = []
    for series_or_df, label in zip(data, moment_set_labels):
        # Join index levels for MultiIndex objects.
        if isinstance(series_or_df.index, pd.MultiIndex):
            series_or_df = series_or_df.rename(index=str)
            series_or_df.index = series_or_df.index.to_flat_index().str.join("_")
        # If moments are a pandas.Series, convert into pandas.DataFrame.
        if isinstance(series_or_df, pd.Series):
            # Unnamed pandas.Series receive a name based on a counter.
            if series_or_df.name is None:
                series_or_df = series_or_df.to_frame(name=next(counter))
            else:
                series_or_df = series_or_df.to_frame()

        # Create pandas.DataFrame in tidy format.
        tidy_df = series_or_df.unstack()
        tidy_df.index.names = ("moment_column", "moment_index")
        tidy_df.rename("value", inplace=True)
        tidy_df = tidy_df.reset_index()
        tidy_df["moment_set"] = label
        tidy_data.append(tidy_df)

    return pd.concat(tidy_data, ignore_index=True)


def _reconstruct_inputs(inputs, dict_keys=None):
    """Reconstruct inputs from lists back to a dictionary or single object."""
    if dict_keys is not None:
        output = dict(zip(dict_keys, inputs))
    elif len(inputs) == 1:
        output = inputs[0]
    else:
        output = inputs

    return output
