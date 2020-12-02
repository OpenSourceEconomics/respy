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
    empirical_moments : pandas.DataFrame or pandas.Series or dict or list
        Contains the empirical moments calculated for the observed data. Moments
        should be saved to pandas.DataFrame or pandas.Series that can either be
        passed to the function directly or as items of a list or dictionary.
        Index of pandas.DataFrames can be of type MultiIndex, but columns cannot.
    weighting_matrix : numpy.ndarray, default None
        Square matrix of dimension (NxN) with N denoting the number of
        empirical_moments. Used to weight squared moment errors. Will use
        identity matrix by default.
    n_simulation_periods : int, default None
        Dictates the number of periods in the simulated dataset.
        This option does not affect ``options["n_periods"]`` which controls the
        number of periods for which decision rules are computed.
    return_scalar : bool, default True
        Indicates whether to return the scalar value of weighted square product of
        moment error vector or dictionary that additionally contains vector of
        (weighted) moment errors, simulated moments that follow the structure
        of empirical moments, and simulated as well as empirical moments in a
        pandas.DataFrame that adheres to a tidy data format. The dictionary will contain
        the following key and value pairs:

        - "value": Scalar vale of weighted moment errors (float)
        - "root_contributions": Moment error vectors multiplied with root of weighting
          matrix (numpy.ndarray)
        - "simulated_moments": Simulated moments for given parametrization. Will be in
        the same data format as `empirical_moments` (pandas.Series or pandas.DataFrame
        or list or dict)
        - "comparison_plot_data": A :class:`pandas.DataFrame` that contains both
        empirical and simulated moments in a tidy data format (pandas.DataFrame). Data
        contains the following columns:

            - ``moment_column``: Contains the column names of the moment
            DataFrames/Series names.
            - ``moment_index``: Contains the index of the moment DataFrames/
            Series.MultiIndex indices will be joined to one string.
            - ``value``: Contains moment values.
            - ``moment_set``: Indicator for each set of moments, will use keys if
            empirical_moments are specified in a dict. Moments input as lists will be
            numbered according to position.
            - ``kind``: Indicates whether moments are empirical or simulated.

    Returns
    -------
    moment_errors_func : callable
         Function where all arguments except the parameter vector are set.

    Raises
    ------
    ValueError
        If replacement function cannot be broadcast (1:1 or 1:N) to simulated moments.
    ValueError
        If the number of functions to compute the simulated moments does not match the
        number of empirical moments.

    """
    empirical_moments = copy.deepcopy(empirical_moments)
    are_empirical_moments_dict = isinstance(empirical_moments, dict)

    if weighting_matrix is None:
        weighting_matrix = get_diag_weighting_matrix(empirical_moments)

    simulate = get_simulate_func(
        params=params, options=options, n_simulation_periods=n_simulation_periods
    )

    empirical_moments = _harmonize_input(empirical_moments)
    calc_moments = _harmonize_input(calc_moments)

    # If only one replacement function is given for multiple sets of moments,
    # duplicate replacement function for all sets of simulated moments.
    if replace_nans is None:
        replace_nans = _return_input

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

    moment_errors_func = functools.partial(
        moment_errors,
        simulate=simulate,
        calc_moments=calc_moments,
        replace_nans=replace_nans,
        empirical_moments=empirical_moments,
        weighting_matrix=weighting_matrix,
        return_scalar=return_scalar,
        are_empirical_moments_dict=are_empirical_moments_dict,
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
    are_empirical_moments_dict,
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
    replace_nans : dict
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
        Indicates whether to return the scalar value of weighted square product of
        moment error vector or dictionary that additionally contains vector of
        (root weighted) moment errors, simulated moments that follow the structure
        of empirical moments, and simulated as well as empirical moments in a
        pandas.DataFrame that adheres to a tidy data format.
    are_empirical_moments_dict : bool
        Indicates whether empirical_moments are originally saved to a dict. Used
        for return of simulated moments in the same form when return_scalar is False.

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

    simulated_moments = {
        name: replace_nans[name](sim_mom) for name, sim_mom in simulated_moments.items()
    }

    flat_empirical_moments = _flatten_index(empirical_moments)
    flat_simulated_moments = _flatten_index(simulated_moments)

    moment_errors = flat_empirical_moments - flat_simulated_moments

    # Return only scalar value or dictionary with additional information:
    # weighted moment errors, simulated moments, and empirical and simulated moments
    # in a DataFrame that adheres to a tidy data format.
    out = moment_errors.T @ weighting_matrix @ moment_errors

    if not return_scalar:
        if not are_empirical_moments_dict:
            simulated_moments = _reconstruct_input_from_dict(simulated_moments)

        out = {
            "value": out,
            "root_contributions": pd.Series(
                moment_errors @ np.sqrt(weighting_matrix), index=moment_errors.index
            ),
            "simulated_moments": _reconstruct_input_from_dict(simulated_moments),
            "comparison_plot_data": _create_comparison_plot_data_msm(
                empirical_moments, simulated_moments
            ),
        }

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


def _harmonize_input(x):
    """Harmonize different types of inputs by turning all inputs into dicts.

    - pandas.DataFrames/Series and callable functions will turn into a dict containing a
      single item (i.e. the input).
    - Dictionaries will be left as is.

    Parameters
    ----------
    x : pandas.DataFrame or pandas.Series or callable or list or dict

    Returns
    -------
    x : dict

    """
    # Convert single pandas.DataFrames, pandas.Series or function into dict containing
    # one item.
    if isinstance(x, (pd.DataFrame, pd.Series)) or callable(x):
        x = {"0": x}

    # Turn lists into dictionary.
    elif isinstance(x, list):
        x = {str(i): x_ for i, x_ in enumerate(x)}

    elif isinstance(x, dict):
        pass

    else:
        raise TypeError(
            "Function only accepts lists, dictionaries, functions, pandas.Series and "
            "pandas.DataFrames as inputs."
        )

    return x


def _flatten_index(moments):
    """Flatten the index as a combination of the former index and the columns.

    Parameters
    ----------
    moments : dict

    Returns
    -------
    pandas.DataFrame

    """
    moments = copy.deepcopy(moments)
    data_flat = []

    for name, series_or_df in moments.items():
        series_or_df.index = series_or_df.index.map(str)
        # Unstack pandas.DataFrames and pandas.Series to add
        # columns/name to index.
        if isinstance(series_or_df, pd.DataFrame):
            df = series_or_df.rename(columns=lambda x: f"{name}_{x}")
        # pandas.Series without a name are named using a counter to avoid duplicate
        # indexes.
        elif isinstance(series_or_df, pd.Series):
            df = series_or_df.to_frame(name=name)
        else:
            raise NotImplementedError

        # Columns to the index.
        df = df.unstack()
        df.index = df.index.to_flat_index().str.join("_")
        data_flat.append(df)

    return pd.concat(data_flat)


def _create_comparison_plot_data_msm(empirical_moments, simulated_moments):
    """Create pandas.DataFrame for estimagic comparison plots.

    Returned object contains empirical and simulated moments.

    Parameters
    ----------
    empirical_moments : dict
    simulated_moments : dict

    Returns
    -------
    pandas.DataFrame

    """
    tidy_empirical_moments = _create_tidy_data(empirical_moments)
    tidy_simulated_moments = _create_tidy_data(simulated_moments)

    tidy_empirical_moments["kind"] = "empirical"
    tidy_simulated_moments["kind"] = "simulated"

    df = pd.concat([tidy_empirical_moments, tidy_simulated_moments], ignore_index=True)

    df[["moment_set", "kind"]] = df[["moment_set", "kind"]].astype("category")

    return df


def _create_tidy_data(moments):
    """Create tidy data from dict containing pandas.DataFrames and/or pandas.Series.

    Parameters
    ----------
    moments : dict

    Returns
    -------
    pandas.DataFrame

    """
    tidy_data = []
    for name, series_or_df in moments.items():
        # Join index levels for MultiIndex objects.
        if isinstance(series_or_df.index, pd.MultiIndex):
            series_or_df = series_or_df.rename(index=str)
            series_or_df.index = series_or_df.index.to_flat_index().str.join("_")
        # If moments are a pandas.Series, convert into pandas.DataFrame.
        if isinstance(series_or_df, pd.Series):
            # Unnamed pandas.Series receive a name based on a counter.
            series_or_df = series_or_df.to_frame(name=name)

        # Create pandas.DataFrame in tidy format.
        tidy_df = series_or_df.unstack()
        tidy_df.index.names = ("moment_column", "moment_index")
        tidy_df.rename("value", inplace=True)
        tidy_df = tidy_df.reset_index()
        tidy_df["moment_set"] = name
        tidy_data.append(tidy_df)

    return pd.concat(tidy_data, ignore_index=True)


def _reconstruct_input_from_dict(x):
    """Reconstruct input from dict back to a list or single object.

    Parameters
    ----------
    x : dict

    Returns
    -------
    out : pandas.DataFrame or pandas.Series or callable or list

    """
    out = list(x.values())

    if len(out) == 1:
        out = out[0]

    return out


def _return_input(x):
    return x
