""" Module that contains the class that carries around information for the
    ROBUPY package
"""

# standard library
import pandas as pd
import numpy as np

# project library
from robupy.clsMeta import MetaCls


class RobupyCls(MetaCls):
    
    def __init__(self):
        """ Initialization of hand-crafted class for package management.
        """
        self.attr = dict()

        self.attr['init_dict'] = None

        # Derived attributes
        self.attr['seed_simulation'] = None

        self.attr['seed_solution'] = None

        self.attr['is_ambiguous'] = None

        self.attr['eps_cholesky'] = None

        self.attr['num_periods'] = None

        self.attr['num_agents'] = None

        self.attr['num_draws'] = None

        self.attr['is_python'] = None

        self.attr['edu_start'] = None

        self.attr['is_debug'] = None

        self.attr['edu_max'] = None

        self.attr['shocks'] = None

        self.attr['version'] = None

        self.attr['delta'] = None

        self.attr['store'] = None

        # Auxiliary object
        self.attr['min_idx'] = None

        # Ambiguity
        self.attr['measure'] = None

        self.attr['level'] = None

        # Results
        self.attr['periods_payoffs_ex_ante'] = None

        self.attr['states_number_period'] = None

        self.attr['mapping_state_idx'] = None

        self.attr['periods_emax'] = None

        self.attr['states_all'] = None

        self.attr['is_solved'] = False

        # The ex post realizations are only stored for debugging purposes.
        # In the special case of no randomness, they have to be equal to the
        # ex ante version. The same is true for the future payoffs
        self.attr['periods_payoffs_ex_post'] = None

        self.attr['periods_future_payoffs'] = None

        # Status indicator
        self.is_locked = False

        self.is_first = True

    ''' Derived attributes
    '''
    def _derived_attributes(self):
        """ Calculate derived attributes.
        """
        # Distribute class attributes
        init_dict = self.attr['init_dict']

        is_first = self.is_first

        # Extract information from initialization dictionary and construct
        # auxiliary objects.
        if is_first:

            self.attr['seed_simulation'] = init_dict['SIMULATION']['seed']

            self.attr['num_agents'] = init_dict['SIMULATION']['agents']

            self.attr['seed_solution'] = init_dict['SOLUTION']['seed']

            self.attr['num_periods'] = init_dict['BASICS']['periods']

            self.attr['measure'] = init_dict['AMBIGUITY']['measure']

            self.attr['edu_start'] = init_dict['EDUCATION']['start']

            self.attr['num_draws'] = init_dict['SOLUTION']['draws']

            self.attr['version'] = init_dict['PROGRAM']['version']

            self.attr['is_debug'] = init_dict['PROGRAM']['debug']

            self.attr['level'] = init_dict['AMBIGUITY']['level']

            self.attr['edu_max'] = init_dict['EDUCATION']['max']

            self.attr['store'] = init_dict['SOLUTION']['store']

            self.attr['delta'] = init_dict['BASICS']['delta']

            self.attr['shocks'] = init_dict['SHOCKS']

            # Auxiliary objects
            num_periods = self.attr['num_periods']

            edu_start = self.attr['edu_start']

            edu_max = self.attr['edu_max']

            shocks = self.attr['shocks']

            self.attr['min_idx'] = min(num_periods, (edu_max - edu_start + 1))

            self.attr['eps_zero'] = (np.count_nonzero(shocks) == 0)

            if self.attr['eps_zero']:
                self.attr['eps_cholesky'] = np.zeros((4, 4))
            else:
                self.attr['eps_cholesky'] = np.linalg.cholesky(shocks)

            self.attr['is_ambiguous'] = (self.attr['level'] > 0.00)

            self.attr['is_python'] = (self.attr['version'] == 'PYTHON')

            # Update status indicator
            self.is_first = False

    def _check_integrity(self):
        """ Check integrity of class instance. This testing is done the first
        time the class is locked and if the package is running in debug mode.
        """
        # Check applicability
        if (not self.is_first) and (not self.attr['is_debug']):
            return

        # Distribute class attributes
        seed_simulation = self.attr['seed_simulation']

        seed_solution = self.attr['seed_solution']

        is_ambiguous = self.attr['is_ambiguous']

        num_agents = self.attr['num_agents']

        num_periods = self.attr['num_periods']

        edu_start = self.attr['edu_start']

        is_python = self.attr['is_python']

        num_draws = self.attr['num_draws']

        is_debug = self.attr['is_debug']

        eps_zero = self.attr['eps_zero']

        measure = self.attr['measure']

        edu_max = self.attr['edu_max']

        shocks = self.attr['shocks']

        delta = self.attr['delta']

        level = self.attr['level']

        version = self.attr['version']

        is_first = self.is_first

        # Debug status
        assert (is_debug in [True, False])

        # Ambiguity in environment
        assert (is_ambiguous in [True, False])

        # Version of implementation
        assert (is_python in [True, False])

        # Constraints
        if is_ambiguous and version in ['F2PY', 'FORTRAN']:
            assert (measure in ['kl'])
        if is_ambiguous:
            assert (eps_zero is False)

        # Seeds
        for seed in [seed_solution, seed_simulation]:
            assert (np.isfinite(seed))
            assert (isinstance(seed, int))
            assert (seed > 0)

        # First
        assert (is_first in [True, False])

        # Number of agents
        assert (np.isfinite(num_agents))
        assert (isinstance(num_agents, int))
        assert (num_agents > 0)

        # Number of periods
        assert (np.isfinite(num_periods))
        assert (isinstance(num_periods, int))
        assert (num_periods > 0)

        # Measure for ambiguity
        assert (measure in ['kl', 'absolute'])

        # Start of education level
        assert (np.isfinite(edu_start))
        assert (isinstance(edu_start, int))
        assert (edu_start >= 0)

        # Number of draws for Monte Carlo integration
        assert (np.isfinite(num_draws))
        assert (isinstance(num_draws, int))
        assert (num_draws >= 0)

        # Level of ambiguity
        assert (np.isfinite(level))
        assert (isinstance(level, float))
        assert (level >= 0.00)

        # Maximum level of education
        assert (np.isfinite(edu_max))
        assert (isinstance(edu_max, int))
        assert (edu_max >= 0)
        assert (edu_max >= edu_start)

        # Debugging mode
        assert (is_debug in [True, False])

        # Discount factor
        assert (np.isfinite(delta))
        assert (isinstance(delta, float))
        assert (delta >= 0.00)

        # Version version of package
        assert (version in ['FORTRAN', 'F2PY', 'PYTHON'])

        # Shock distribution
        assert (isinstance(shocks, np.ndarray))
        assert (np.all(np.isfinite(shocks)))
        assert (shocks.shape == (4, 4))

        # Check integrity of results as well
        self._check_integrity_results()

    def _check_integrity_results(self):
        """ This methods check the integrity of the results.
        """

        # Distribute auxiliary objects
        num_periods = self.attr['num_periods']

        edu_start = self.attr['edu_start']

        edu_max = self.attr['edu_max']

        # Distribute results
        periods_payoffs_ex_ante = self.attr['periods_payoffs_ex_ante']

        periods_future_payoffs = self.attr['periods_future_payoffs']

        states_number_period = self.attr['states_number_period']

        mapping_state_idx = self.attr['mapping_state_idx']

        periods_emax = self.attr['periods_emax']

        states_all = self.attr['states_all']

        # Check the creation of the state space
        is_applicable = (states_all is not None)
        is_applicable = is_applicable and (states_number_period is not None)
        is_applicable = is_applicable and (mapping_state_idx is not None)

        if is_applicable:
            # If the agent never increased their level of education, the lagged
            # education variable cannot take a value larger than zero.
            for period in range(1, num_periods):
                indices = (np.where(states_all[period, :, :][:, 2] == 0))
                for index in indices:
                    assert (np.all(states_all[period, :, :][index, 3]) == 0)

            # No values can be larger than constraint time. The exception in the
            # lagged schooling variable in the first period, which takes value
            # one but has index zero.
            for period in range(num_periods):
                assert (np.nanmax(states_all[period, :, :3]) <= period)

            # Lagged schooling can only take value zero or one if finite.
            # In fact, it can only take value one in the first period.
            for period in range(num_periods):
                assert (np.all(states_all[0, :, 3]) == 1)
                assert (np.nanmax(states_all[period, :, 3]) == 1)
                assert (np.nanmin(states_all[period, :, :3]) == 0)

            # All finite values have to be larger or equal to zero. The loop is
            # required as np.all evaluates to FALSE for this condition
            # (see NUMPY documentation).
            for period in range(num_periods):
                assert (
                    np.all(states_all[period, :states_number_period[period]] >= 0))

            # The maximum number of additional education years is never larger
            # than (EDU_MAX - EDU_START).
            for period in range(num_periods):
                assert (np.nanmax(states_all[period, :, :][:, 2], axis=0) <= (
                    edu_max - edu_start))

            # Check for duplicate rows in each period
            for period in range(num_periods):
                assert (np.sum(pd.DataFrame(
                    states_all[period, :states_number_period[period],
                    :]).duplicated()) == 0)

            # Checking validity of state space values. All valid
            # values need to be finite.
            for period in range(num_periods):
                assert (np.all(
                    np.isfinite(states_all[period, :states_number_period[period]])))

            # There are no infinite values in final period.
            assert (np.all(np.isfinite(states_all[(num_periods - 1), :, :])))

            # There are is only one finite realization in period one.
            assert (np.sum(np.isfinite(mapping_state_idx[0, :, :, :, :])) == 1)

            # If valid, the number of state space realizations in period two is
            # four.
            if num_periods > 1:
                assert (np.sum(np.isfinite(mapping_state_idx[1, :, :, :, :])) == 4)

            # Check that mapping is defined for all possible realizations of the
            # state space by period. Check that mapping is not defined for all
            # inadmissible values.
            is_infinite = np.tile(False, reps=mapping_state_idx.shape)
            for period in range(num_periods):
                # Subsetting valid indices
                indices = states_all[period, :states_number_period[period], :]
                for index in indices:
                    # Check for finite value at admissible state
                    assert (np.isfinite(mapping_state_idx[
                                            period, index[0], index[1], index[2],
                                            index[3]]))
                    # Record finite value
                    is_infinite[
                        period, index[0], index[1], index[2], index[3]] = True
            # Check that all admissible states are finite
            assert (np.all(np.isfinite(mapping_state_idx[is_infinite == True])))

            # Check that all inadmissible states are infinite
            assert (
                np.all(np.isfinite(mapping_state_idx[is_infinite == False])) == False)

        # Check the calculated ex ante payoffs
        is_applicable = (states_all is not None)
        is_applicable = is_applicable and (states_number_period is not None)
        is_applicable = is_applicable and (periods_payoffs_ex_ante is not None)

        if is_applicable:
            # Check that the payoffs are finite for all admissible values and
            # infinite for all others.
            is_infinite = np.tile(False, reps=periods_payoffs_ex_ante.shape)
            for period in range(num_periods):
                # Loop over all possible states
                for k in range(states_number_period[period]):
                    # Check that wages are all positive
                    assert (np.all(periods_payoffs_ex_ante[period, k, :2] > 0.0))
                    # Check for finite value at admissible state
                    assert (
                        np.all(np.isfinite(periods_payoffs_ex_ante[period, k, :])))
                    # Record finite value
                    is_infinite[period, k, :] = True
                # Check that all admissible states are finite
                assert (
                    np.all(np.isfinite(periods_payoffs_ex_ante[is_infinite ==
                                                               True])))
                # Check that all inadmissible states are infinite
                if num_periods > 1:
                    assert (np.all(np.isfinite(
                        periods_payoffs_ex_ante[is_infinite == False])) == False)

        # Check the expected future value
        is_applicable = (periods_emax is not None)
        is_applicable = is_applicable and (periods_future_payoffs is not None)

        if is_applicable:
            # Check that the payoffs are finite for all admissible values and
            # infinite for all others.
            is_infinite = np.tile(False, reps=periods_emax.shape)
            for period in range(num_periods):
                # Loop over all possible states
                for k in range(states_number_period[period]):
                    # Check for finite value at admissible state
                    assert (np.all(np.isfinite(periods_emax[period, k])))
                    # Record finite value
                    is_infinite[period, k] = True
                # Check that all admissible states are finite
                assert (np.all(np.isfinite(periods_emax[is_infinite == True])))
                # Check that all inadmissible states are infinite
                if num_periods == 1:
                    assert (len(periods_emax[is_infinite == False]) == 0)
                else:
                    assert (
                        np.all(np.isfinite(periods_emax[is_infinite == False])) == False)

            # Check that the payoffs are finite for all admissible values and
            # infinite for all others.
            for period in range(num_periods - 1):
                # Loop over all possible states
                for k in range(states_number_period[period]):
                    # Check for finite value at admissible state, infinite
                    # values are allowed for the third column when the
                    # maximum level of education is attained.
                    assert (np.all(np.isfinite(periods_future_payoffs[period, k, :2])))
                    assert (np.all(np.isfinite(periods_future_payoffs[period, k, 3])))
                    # Special checks for infinite value due to
                    # high education.
                    if not np.isfinite(periods_future_payoffs[period, k, 2]):
                        assert (states_all[period, k][2] == edu_max - edu_start)
