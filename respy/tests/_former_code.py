import numpy as np
from numba import njit


@njit
def _create_state_space_kw94(n_periods, n_types, edu_starts, edu_max):
    """Create the state space.

    The state space consists of all admissible combinations of the following elements:
    period, experience in OCCUPATION A, experience in OCCUPATION B, years of schooling,
    the lagged choice and the type of the agent.

    The major problem which is solved here is that the natural representation of the
    state space is a graph where each parent node is connected with its children's
    nodes. This structure makes it easy to traverse the graph. In contrast to that, for
    many subsequent calculations, e.g. generating the covariates for each state, a
    tabular format has better performance, but looses the information on the
    connections.

    The current implementation of ``states`` and ``indexer`` allows to have both
    advantages at the cost of an additional object. ``states`` stores the information on
    states in a tabular format. ``indexer`` is a matrix where each characteristic of the
    state space represents one dimension. The values of the matrix are the indices of
    states in ``states``. Traversing the state space is as easy as incrementing the
    right indices of ``indexer`` by 1 and use the resulting index in ``states``.

    Parameters
    ----------
    n_periods : int
        Number of periods in the state space.
    n_types : int
        Number of types of agents.
    edu_starts : numpy.ndarray
        Contains levels of initial education.
    edu_max : int
        Maximum level of education which can be obtained by an agent.

    Returns
    -------
    states : numpy.ndarray
        Array with shape (n_states, 6) containing period, experience in OCCUPATION A,
        experience in OCCUPATION B, years of schooling, the lagged choice and the type
        of the agent.
    indexer : numpy.ndarray
        A matrix where each dimension represents a characteristic of the state space.
        Switching from one state is possible via incrementing appropriate indices by 1.

    Examples
    --------
    >>> n_periods = 40
    >>> n_types = 1
    >>> edu_starts, edu_max = np.array([10]), 20
    >>> states, indexer = _create_state_space_kw94(
    ...     n_periods, n_types, edu_starts, edu_max
    ... )
    >>> states.shape
    (317367, 6)
    >>> len(indexer)
    40
    >>> indexer[-1].shape
    (40, 40, 21, 4, 1)

    """
    data = []
    indexer = []

    # Initialize counter
    i = 0

    # Construct state space by periods
    for period in range(n_periods):

        # Build periodic indexer.
        max_edu_starts = max(edu_starts)
        dim_edu = min(max_edu_starts + period, edu_max) + 1
        shape = (period + 1, period + 1, dim_edu, 4, n_types)
        sub_indexer = np.full(shape, -1, dtype=np.int32)
        indexer.append(sub_indexer)

        # Loop over all unobserved types
        for type_ in range(n_types):

            # Loop overall all initial levels of schooling
            for edu_start in edu_starts:

                # For occupations and education it is necessary to loop over period
                # + 1 as zero has to be included if it is never this choice and period
                # + 1 if it is always the same choice.

                # Furthermore, the time constraint of agents is always fulfilled as
                # previous choices limit the space for subsequent choices.

                # Loop over all admissible work experiences for Occupation A
                for exp_a in range(period + 1):

                    # Loop over all admissible work experience for Occupation B
                    for exp_b in range(period + 1 - exp_a):

                        # Loop over all admissible additional education levels
                        for edu_add in range(
                            min(period + 1 - exp_a - exp_b, edu_max + 1 - edu_start)
                        ):

                            # Loop over all admissible values for the lagged activity:
                            # (1) Occupation A, (2) Occupation B, (3) Education, and (4)
                            # Home.
                            for lagged_choice in range(4):

                                if period > 0:

                                    # (0, 1) Whenever an agent has only worked in
                                    # Occupation A, then the lagged choice cannot be
                                    # anything other than one.
                                    if lagged_choice != 0 and exp_a == period:
                                        continue

                                    # (0, 2) Whenever an agent has only worked in
                                    # Occupation B, then the lagged choice cannot be
                                    # anything other than two
                                    if lagged_choice != 1 and exp_b == period:
                                        continue

                                    # (0, 3) Whenever an agent has only acquired
                                    # additional education, then the lagged choice
                                    # cannot be anything other than three.
                                    if lagged_choice != 2 and edu_add == period:
                                        continue

                                    # (0, 4) Whenever an agent has not acquired any
                                    # additional education and we are not in the first
                                    # period, then lagged activity cannot take a value
                                    # of three.
                                    if lagged_choice == 2 and edu_add == 0:
                                        continue

                                    # (0, 5) Whenever an agent has always chosen
                                    # Occupation A, Occupation B or education, then
                                    # lagged activity cannot take a value of four.
                                    if (
                                        lagged_choice == 3
                                        and exp_a + exp_b + edu_add == period
                                    ):
                                        continue

                                # (2, 1) An individual that has never worked in
                                # Occupation A cannot have that lagged activity.
                                if lagged_choice == 0 and exp_a == 0:
                                    continue

                                # (3, 1) An individual that has never worked in
                                # Occupation B cannot have a that lagged activity.
                                if lagged_choice == 1 and exp_b == 0:
                                    continue

                                # (1, 1) In the first period individual either were in
                                # school the previous period as well or at home. They
                                # cannot have any work experience.
                                if period == 0:
                                    if lagged_choice in [0, 1]:
                                        continue

                                # Continue if state still exist. This condition is only
                                # triggered by multiple initial levels of education.
                                if (
                                    indexer[period][
                                        exp_a,
                                        exp_b,
                                        edu_start + edu_add,
                                        lagged_choice,
                                        type_,
                                    ]
                                    != -1
                                ):
                                    continue

                                # Collect mapping of state space to array index.
                                indexer[period][
                                    exp_a,
                                    exp_b,
                                    edu_start + edu_add,
                                    lagged_choice,
                                    type_,
                                ] = i

                                i += 1

                                # Store information in a dictionary and append to data.
                                row = (
                                    period,
                                    exp_a,
                                    exp_b,
                                    edu_start + edu_add,
                                    lagged_choice,
                                    type_,
                                )
                                data.append(row)

    states = np.array(data)

    return states, indexer


@njit
def _create_state_space_kw97_base(n_periods, n_types, edu_starts, edu_max):
    """Create the state space for the base model of Keane and Wolpin (1997).

    Examples
    --------
    >>> n_periods = 40
    >>> n_types = 1
    >>> edu_starts, edu_max = np.array([10]), 20
    >>> states, indexer = _create_state_space_kw97_base(
    ...     n_periods, n_types, edu_starts, edu_max
    ... )
    >>> states.shape
    (848672, 6)
    >>> len(indexer)
    40
    >>> indexer[-1].shape
    (40, 40, 40, 21, 1)

    """
    data = []
    indexer = []

    # Initialize counter
    i = 0

    # Construct state space by periods
    for period in range(n_periods):

        # Build periodic indexer.
        max_edu_starts = max(edu_starts)
        dim_edu = min(max_edu_starts + period, edu_max) + 1
        shape = (period + 1, period + 1, period + 1, dim_edu, n_types)
        sub_indexer = np.full(shape, -1, dtype=np.int32)
        indexer.append(sub_indexer)

        # Loop over all unobserved types
        for type_ in range(n_types):

            # Loop overall all initial levels of schooling
            for edu_start in edu_starts:

                # For occupations and education it is necessary to loop over period
                # + 1 as zero has to be included if it is never this choice and period
                # + 1 if it is always the same choice.

                # Furthermore, the time constraint of agents is always fulfilled as
                # previous choices limit the space for subsequent choices.

                # Loop over all admissible work experiences for Occupation A
                for exp_a in range(period + 1):

                    # Loop over all admissible work experience for Occupation B
                    for exp_b in range(period + 1 - exp_a):

                        # Loop over all admissible work experience for Occupation B
                        for exp_mil in range(period + 1 - exp_a - exp_b):

                            # Loop over all admissible additional education levels
                            for edu_add in range(
                                min(
                                    period + 1 - exp_a - exp_b - exp_mil,
                                    edu_max + 1 - edu_start,
                                )
                            ):

                                # Continue if state still exist. This condition is
                                # only triggered by multiple initial levels of
                                # education.
                                if (
                                    indexer[period][
                                        exp_a,
                                        exp_b,
                                        exp_mil,
                                        edu_start + edu_add,
                                        type_,
                                    ]
                                    != -1
                                ):
                                    continue

                                # Collect mapping of state space to array index.
                                indexer[period][
                                    exp_a, exp_b, exp_mil, edu_start + edu_add, type_
                                ] = i

                                i += 1

                                # Store information in a dictionary and append to
                                # data.
                                row = (
                                    period,
                                    exp_a,
                                    exp_b,
                                    exp_mil,
                                    edu_start + edu_add,
                                    type_,
                                )
                                data.append(row)

    states = np.array(data)

    return states, indexer


@njit
def _create_state_space_kw97_extended(n_periods, n_types, edu_starts, edu_max):
    """Create the state space for the extended model of Keane and Wolpin (1997).

    Examples
    --------
    >>> n_periods = 40
    >>> n_types = 1
    >>> edu_starts, edu_max = np.array([10]), 20
    >>> states, indexer = _create_state_space_kw97_extended(
    ...     n_periods, n_types, edu_starts, edu_max
    ... )
    >>> states.shape
    (3770152, 7)
    >>> len(indexer)
    40
    >>> indexer[-1].shape
    (40, 40, 40, 21, 5, 1)

    """
    data = []
    indexer = []

    # Initialize counter
    i = 0

    # Construct state space by periods
    for period in range(n_periods):

        # Build periodic indexer.
        max_edu_starts = max(edu_starts)
        dim_edu = min(max_edu_starts + period, edu_max) + 1
        shape = (period + 1, period + 1, period + 1, dim_edu, 5, n_types)
        sub_indexer = np.full(shape, -1, dtype=np.int32)
        indexer.append(sub_indexer)

        # Loop over all unobserved types
        for type_ in range(n_types):

            # Loop overall all initial levels of schooling
            for edu_start in edu_starts:

                # For occupations and education it is necessary to loop over period
                # + 1 as zero has to be included if it is never this choice and period
                # + 1 if it is always the same choice.

                # Furthermore, the time constraint of agents is always fulfilled as
                # previous choices limit the space for subsequent choices.

                # Loop over all admissible work experiences for Occupation A
                for exp_a in range(period + 1):

                    # Loop over all admissible work experience for Occupation B
                    for exp_b in range(period + 1 - exp_a):

                        # Loop over all admissible work experience for Occupation B
                        for exp_mil in range(period + 1 - exp_a - exp_b):

                            # Loop over all admissible additional education levels
                            for edu_add in range(
                                min(
                                    period + 1 - exp_a - exp_b - exp_mil,
                                    edu_max + 1 - edu_start,
                                )
                            ):

                                # Loop over all admissible values for the lagged
                                # activity: (1) Occupation A, (2) Occupation B, (3)
                                # Military, (4) Education, and (5) Home.
                                for lagged_choice in range(5):

                                    if period > 0:

                                        # (0, 1) Whenever an agent has only worked in
                                        # Occupation A, then the lagged choice cannot be
                                        # anything other than one.
                                        if lagged_choice != 0 and exp_a == period:
                                            continue

                                        # (0, 2) Whenever an agent has only worked in
                                        # Occupation B, then the lagged choice cannot be
                                        # anything other than two
                                        if lagged_choice != 1 and exp_b == period:
                                            continue

                                        # (0, 3) Whenever an agent has only worked in
                                        # Military, then the lagged choice cannot be
                                        # anything other than two
                                        if lagged_choice != 2 and exp_mil == period:
                                            continue

                                        # (0, 3) Whenever an agent has only acquired
                                        # additional education, then the lagged choice
                                        # cannot be anything other than three.
                                        if lagged_choice != 3 and edu_add == period:
                                            continue

                                        # (0, 4) Whenever an agent has not acquired any
                                        # additional education and we are not in the
                                        # first period, then lagged activity cannot take
                                        # a value of three.
                                        if lagged_choice == 3 and edu_add == 0:
                                            continue

                                        # (0, 5) Whenever an agent has always chosen
                                        # Occupation A, Occupation B, Military or
                                        # education, then lagged activity cannot take a
                                        # value of four.
                                        if (
                                            lagged_choice == 4
                                            and exp_a + exp_b + exp_mil + edu_add
                                            == period
                                        ):
                                            continue

                                    # (2, 1) An individual that has never worked in
                                    # Occupation A cannot have that lagged activity.
                                    if lagged_choice == 0 and exp_a == 0:
                                        continue

                                    # (3, 1) An individual that has never worked in
                                    # Occupation B cannot have a that lagged activity.
                                    if lagged_choice == 1 and exp_b == 0:
                                        continue

                                    # (4, 1) An individual that has never worked in
                                    # Military cannot have a that lagged activity.
                                    if lagged_choice == 2 and exp_mil == 0:
                                        continue

                                    # (1, 1) In the first period individuals either were
                                    # in school or at home.
                                    if period == 0:
                                        if lagged_choice in [0, 1, 2]:
                                            continue

                                    # Continue if state still exist. This condition is
                                    # only triggered by multiple initial levels of
                                    # education.
                                    if (
                                        indexer[period][
                                            exp_a,
                                            exp_b,
                                            exp_mil,
                                            edu_start + edu_add,
                                            lagged_choice,
                                            type_,
                                        ]
                                        != -1
                                    ):
                                        continue

                                    # Collect mapping of state space to array index.
                                    indexer[period][
                                        exp_a,
                                        exp_b,
                                        exp_mil,
                                        edu_start + edu_add,
                                        lagged_choice,
                                        type_,
                                    ] = i

                                    i += 1

                                    # Store information in a dictionary and append to
                                    # data.
                                    row = (
                                        period,
                                        exp_a,
                                        exp_b,
                                        exp_mil,
                                        edu_start + edu_add,
                                        lagged_choice,
                                        type_,
                                    )
                                    data.append(row)

    states = np.array(data)

    return states, indexer
