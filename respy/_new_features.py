import itertools

import numpy as np
import pandas as pd


def _create_state_space(
    n_periods, maximum_exp, minimum_exp, n_nonexp_choices, n_types, filters
):
    """Create the state space.

    This process involves two steps. First, the core state space is created which
    abstracts from different initial experiences and instead uses the minimum initial
    experience per choice.

    Secondly, the state space is adjusted by all combinations of initial experiences and
    also filtered, excluding invalid states.

    """
    df = _create_core_state_space(
        n_periods, maximum_exp, minimum_exp, n_nonexp_choices, n_types
    )

    df = _adjust_state_space_with_different_starting_experiences(
        df, minimum_exp, maximum_exp, n_nonexp_choices, filters
    )

    indexer = _create_indexer(df, n_periods, n_nonexp_choices, n_types, maximum_exp)

    return df, indexer


def _create_core_state_space(
    n_periods, maximum_exp, minimum_exp, n_nonexp_choices, n_types
):
    """Create the core state space.

    The core state space is created using the

    """
    maximum_exp = np.array([n_periods if i == -1 else i for i in maximum_exp])
    min_min_exp = np.array([np.min(i) for i in minimum_exp])
    additional_exp = maximum_exp - np.array(min_min_exp)

    # TODO: This process can be multi-processed.
    data = []
    for period in range(n_periods):
        data += list(
            _create_core_state_space_per_period(
                period, additional_exp, n_nonexp_choices, n_types
            )
        )

    df = pd.DataFrame.from_records(
        data, columns=["period", "exp_0", "exp_1", "exp_2", "lagged_choice", "type"]
    )

    return df


def _create_core_state_space_per_period(
    period, additional_exp, n_nonexp_choices, n_types, experiences=None, pos=0
):
    """Create core state space per period.

    First, this function returns a state combined with all possible lagged choices and
    types.

    Secondly, if there exists a choice with experience in ``additional_exp[pos]``, loop
    over all admissible experiences, update the state and pass it to the same function,
    but moving to the next choice which accumulates experience.

    """
    experiences = [0] * len(additional_exp) if experiences is None else experiences

    # Return experiences combined with lagged choices and types.
    for t in range(n_types):
        for i in range(len(experiences) + n_nonexp_choices):
            yield [period] + experiences + [i, t]

    # Check if there is an additional choice left to start another loop.
    if pos < len(experiences):
        # Upper bound of additional experience is given by the remaining time or the
        # maximum experience which can be accumulated in experience[pos].
        remaining_time = period - np.sum(experiences)
        max_experience = additional_exp[pos]

        # +1 is necessary so that the remaining time or max_experience is exhausted.
        for i in range(min(remaining_time, max_experience) + 1):
            # Update experiences and call the same function with the next choice.
            updated_experiences = experiences.copy()
            updated_experiences[pos] += i
            yield from _create_core_state_space_per_period(
                period,
                additional_exp,
                n_nonexp_choices,
                n_types,
                updated_experiences,
                pos + 1,
            )


def _adjust_state_space_with_different_starting_experiences(
    df, minimum_exp, maximum_exp, n_nonexp_choices, filters
):
    # TODO: Has to move to model processing.
    minimum_exp_ = [[i] if isinstance(i, int) else i for i in minimum_exp]
    # Create combinations of starting values
    starting_combinations = itertools.product(*minimum_exp_)

    exp_cols = df.filter(like="exp_").columns.tolist()

    container = []

    for start_comb in starting_combinations:
        df_ = df.copy()

        # Add initial experiences.
        df_[exp_cols] += start_comb

        # Check that time constraint is still fulfilled.
        sum_experiences = df_[exp_cols].subtract(start_comb).sum(axis="columns")
        df_ = df_.loc[df_.period.ge(sum_experiences)].copy()

        # Check that max_experience is still fulfilled.
        df_ = df_.loc[df_[exp_cols].le(maximum_exp).all(axis="columns")].copy()

        df_ = _filter_state_space(df_, filters, n_nonexp_choices, start_comb)

        container.append(df_)

    df = pd.concat(container, axis="rows", sort=False).drop_duplicates()

    return df


def _filter_state_space(df, filters, n_nonexp_choices, initial_exp):
    col = "_restriction_{}"
    n_exp_choices = len(initial_exp)

    # Restriction counter
    counter = itertools.count()

    for definition in filters:

        # If "{i}" is in definition, loop over choices with experiences.
        if "{i}" in definition:
            for i in range(n_exp_choices):
                df[col.format(next(counter))] = df.eval(definition.format(i=i))

        # If "{j}" is in definition, loop over choices without experiences.
        elif "{j}" in definition:
            for j in range(n_exp_choices, n_exp_choices + n_nonexp_choices):
                df[col.format(next(counter))] = df.eval(definition.format(j=j))

        else:
            df[col.format(next(counter))] = df.eval(definition)

    df = df[~df.filter(like="_restriction_").any(axis="columns")]
    df = df.drop(columns=df.filter(like="_restriction").columns.tolist())

    return df


def _create_indexer(df, n_periods, n_nonexp_choices, n_types, maximum_exp):
    n_exp_choices = maximum_exp.shape[0]
    maximum_exp[2] += 1

    shape = (
        (n_periods,) + tuple(maximum_exp) + (n_exp_choices + n_nonexp_choices, n_types)
    )
    indexer = np.full(shape, -1)

    indexer[
        df.period, df.exp_0, df.exp_1, df.exp_2, df.lagged_choice, df.type
    ] = np.arange(df.shape[0])

    return indexer
