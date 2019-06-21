import numpy as np
from numba import njit


@njit
def create_state_space(n_periods, n_types, edu_starts, edu_max):
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
    edu_starts : List[int]
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
    >>> edu_starts, edu_max = [10], 20
    >>> states, indexer = create_state_space(
    ...     n_periods, n_types, edu_starts, edu_max
    ... )
    >>> states.shape
    (317367, 6)
    >>> indexer.shape
    (40, 40, 40, 21, 4, 1)

    """
    data = []

    shape = (n_periods, n_periods, n_periods, edu_max + 1, 4, n_types)
    indexer = np.full(shape, -1, dtype=np.int32)

    # Initialize counter
    i = 0

    # Construct state space by periods
    for period in range(n_periods):

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
                            for choice_lagged in range(4):

                                if period > 0:

                                    # (0, 1) Whenever an agent has only worked in
                                    # Occupation A, then the lagged choice cannot be
                                    # anything other than one.
                                    if choice_lagged != 0 and exp_a == period:
                                        continue

                                    # (0, 2) Whenever an agent has only worked in
                                    # Occupation B, then the lagged choice cannot be
                                    # anything other than two
                                    if choice_lagged != 1 and exp_b == period:
                                        continue

                                    # (0, 3) Whenever an agent has only acquired
                                    # additional education, then the lagged choice
                                    # cannot be anything other than three.
                                    if choice_lagged != 2 and edu_add == period:
                                        continue

                                    # (0, 4) Whenever an agent has not acquired any
                                    # additional education and we are not in the first
                                    # period, then lagged activity cannot take a value
                                    # of three.
                                    if choice_lagged == 2 and edu_add == 0:
                                        continue

                                    # (0, 5) Whenever an agent has always chosen
                                    # Occupation A, Occupation B or education, then
                                    # lagged activity cannot take a value of four.
                                    if (
                                        choice_lagged == 3
                                        and exp_a + exp_b + edu_add == period
                                    ):
                                        continue

                                # (2, 1) An individual that has never worked in
                                # Occupation A cannot have that lagged activity.
                                if choice_lagged == 0 and exp_a == 0:
                                    continue

                                # (3, 1) An individual that has never worked in
                                # Occupation B cannot have a that lagged activity.
                                if choice_lagged == 1 and exp_b == 0:
                                    continue

                                # (1, 1) In the first period individual either were in
                                # school the previous period as well or at home. They
                                # cannot have any work experience.
                                if period == 0:
                                    if choice_lagged in [0, 1]:
                                        continue

                                # Continue if state still exist. This condition is only
                                # triggered by multiple initial levels of education.
                                if (
                                    indexer[
                                        period,
                                        exp_a,
                                        exp_b,
                                        edu_start + edu_add,
                                        choice_lagged,
                                        type_,
                                    ]
                                    != -1
                                ):
                                    continue

                                # Collect mapping of state space to array index.
                                indexer[
                                    period,
                                    exp_a,
                                    exp_b,
                                    edu_start + edu_add,
                                    choice_lagged,
                                    type_,
                                ] = i

                                i += 1

                                # Store information in a dictionary and append to data.
                                row = (
                                    period,
                                    exp_a,
                                    exp_b,
                                    edu_start + edu_add,
                                    choice_lagged,
                                    type_,
                                )
                                data.append(row)

    states = np.array(data)

    return states, indexer
