""" Module that contains the class that carries around information for the
    ROBUPY package
"""

# standard library
import pickle as pkl
import pandas as pd
import numpy as np

# Special care with derived attributes is required to maintain integrity of
# the class instance.
DERIVED_ATTR = ['is_ambiguous', 'is_python', 'min_idx', 'shocks_zero']


class RobupyCls(object):
    """ This class manages the distribution of the use requests throughout
    the toolbox.
    """
    def __init__(self, init_dict):
        """ Initialization of hand-crafted class for package management.
        """
        self.attr = dict()

        self.attr['init_dict'] = init_dict

        # Derived attributes
        self.attr['is_interpolated'] = None

        self.attr['num_draws_emax'] = None

        self.attr['num_draws_prob'] = None

        self.attr['is_ambiguous'] = None

        self.attr['num_periods'] = None

        self.attr['model_paras'] = None

        self.attr['num_agents'] = None

        self.attr['num_points'] = None

        self.attr['seed_data'] = None

        self.attr['seed_prob'] = None

        self.attr['seed_emax'] = None

        self.attr['is_python'] = None

        self.attr['edu_start'] = None

        self.attr['is_debug'] = None

        self.attr['file_sim'] = None

        self.attr['file_est'] = None

        self.attr['edu_max'] = None

        self.attr['version'] = None

        self.attr['min_idx'] = None

        self.attr['measure'] = None

        self.attr['delta'] = None

        self.attr['store'] = None

        self.attr['level'] = None

        # Results
        self.attr['periods_payoffs_systematic'] = None

        self.attr['states_number_period'] = None

        self.attr['mapping_state_idx'] = None

        self.attr['periods_emax'] = None

        self.attr['states_all'] = None

        self.attr['is_solved'] = False

        # The ex post realizations are only stored for debugging purposes.
        # In the special case of no randomness (with all disturbances equal
        # to zero), they have to be equal to the systematic version. The same
        # is true for the future payoffs
        self.attr['periods_payoffs_ex_post'] = None

        self.attr['periods_payoffs_future'] = None

        # Initialization
        self._update_core_attributes()

        self._update_derived_attributes()

        # Status indicator
        self.is_locked = False

    def update_model_paras(self, coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
            shocks, shocks_cholesky):
        """ Update model parameters.
        """
        # Distribute class attributes
        model_paras = self.attr['model_paras']

        # Update model parametrization
        model_paras['shocks_cholesky'] = shocks_cholesky

        model_paras['coeffs_home'] = coeffs_home

        model_paras['coeffs_edu'] = coeffs_edu

        model_paras['coeffs_a'] = coeffs_a

        model_paras['coeffs_b'] = coeffs_b

        model_paras['shocks'] = shocks

        # Update class attributes
        self.attr['model_paras'] = model_paras

        # Update derived attributes. This is required as the shocks_zero
        # indicator depends on the value of shocks. The latter is modified
        # above.
        self._update_derived_attributes()

    def get_status(self):
        """ Get status of class instance.
        """
        return self.is_locked

    def lock(self):
        """ Lock class instance.
        """
        # Antibugging.
        assert (self.get_status() is False)

        # Checks
        self._check_integrity_attributes()

        self._check_integrity_results()

        # Update class attributes.
        self.is_locked = True

    def unlock(self):
        """ Unlock class instance.
        """
        # Antibugging
        assert (self.get_status() is True)

        # Update class attributes
        self.is_locked = False

    def get_attr(self, key):
        """ Get attributes.
        """
        # Antibugging
        assert (self.get_status() is True)
        assert (self._check_key(key) is True)

        # Finishing
        return self.attr[key]

    def set_attr(self, key, value):
        """ Get attributes.
        """
        # Antibugging
        assert (self.get_status() is False)
        assert (self._check_key(key) is True)

        # Finishing
        self.attr[key] = value

        # Special care is required for the attributes which are derived from
        # other core attributes. These derived attributes cannot be set and
        # are checked after each modification. Also, the model
        # parametrization can only be changed by a special function. The
        # initialization dictionary can only be set initially.
        invalid_attr = DERIVED_ATTR + ['model_paras', 'init_dict']

        if key in invalid_attr:
            raise AssertionError('invalid request')

        # Update derived attributes
        self._update_derived_attributes()

    def store(self, file_name):
        """ Store class instance.
        """
        # Antibugging
        assert (self.get_status() is True)
        assert (isinstance(file_name, str))

        # Store.
        pkl.dump(self, open(file_name, 'wb'))

    ''' Private methods
    '''
    def _update_core_attributes(self):
        """ Calculate derived attributes. This is only called when the class
        is initialized
        """
        # Distribute class attributes
        init_dict = self.attr['init_dict']

        # Extract information from initialization dictionary and construct
        # auxiliary objects.
        self.attr['is_interpolated'] = init_dict['INTERPOLATION']['apply']

        self.attr['num_draws_prob'] = init_dict['ESTIMATION']['draws']

        self.attr['num_points'] = init_dict['INTERPOLATION']['points']

        self.attr['num_draws_emax'] = init_dict['SOLUTION']['draws']

        self.attr['num_agents'] = init_dict['SIMULATION']['agents']

        self.attr['num_periods'] = init_dict['BASICS']['periods']

        self.attr['measure'] = init_dict['AMBIGUITY']['measure']

        self.attr['edu_start'] = init_dict['EDUCATION']['start']

        self.attr['seed_data'] = init_dict['SIMULATION']['seed']

        self.attr['seed_prob'] = init_dict['ESTIMATION']['seed']

        self.attr['file_sim'] = init_dict['SIMULATION']['file']

        self.attr['file_est'] = init_dict['ESTIMATION']['file']

        self.attr['seed_emax'] = init_dict['SOLUTION']['seed']

        self.attr['version'] = init_dict['PROGRAM']['version']

        self.attr['is_debug'] = init_dict['PROGRAM']['debug']

        self.attr['edu_max'] = init_dict['EDUCATION']['max']

        self.attr['level'] = init_dict['AMBIGUITY']['level']

        self.attr['store'] = init_dict['SOLUTION']['store']

        self.attr['delta'] = init_dict['BASICS']['delta']

        # Initialize model parameters
        self.attr['model_paras'] = dict()

        self.attr['model_paras']['coeffs_a'] = [init_dict['A']['int']]
        self.attr['model_paras']['coeffs_a'] += init_dict['A']['coeff']

        self.attr['model_paras']['coeffs_b'] = [init_dict['B']['int']]
        self.attr['model_paras']['coeffs_b'] += init_dict['B']['coeff']

        self.attr['model_paras']['coeffs_edu'] = \
            [init_dict['EDUCATION']['int']]
        self.attr['model_paras']['coeffs_edu'] += \
            init_dict['EDUCATION']['coeff']

        self.attr['model_paras']['coeffs_home'] = \
            [init_dict['HOME']['int']]

        self.attr['model_paras']['shocks'] = init_dict['SHOCKS']

        # Carry the Cholesky decomposition as part of the model
        # parameters.
        shocks = self.attr['model_paras']['shocks']
        if np.count_nonzero(shocks) == 0:
            shocks_cholesky = np.zeros((4, 4))
        else:
            shocks_cholesky = np.linalg.cholesky(shocks)
        self.attr['model_paras']['shocks_cholesky'] = shocks_cholesky

        # Ensure that all elements in the dictionary are of the same
        # type.
        keys = ['coeffs_a', 'coeffs_b', 'coeffs_edu', 'coeffs_home']
        keys += ['shocks']
        for key_ in keys:
            self.attr['model_paras'][key_] = \
                np.array(self.attr['model_paras'][key_])

        # Delete the duplicated information from the initialization
        # dictionary. Special treatment of EDUCATION is required as it
        # contains other information about education than just the
        # payoff parametrization.
        del init_dict['EDUCATION']['int']
        del init_dict['EDUCATION']['coeff']

        for key_ in ['A', 'B', 'HOME', 'SHOCKS']:
            del init_dict[key_]

    def _update_derived_attributes(self):
        """ Update derived attributes.
        """

        # Distribute class attributes
        model_paras = self.attr['model_paras']

        num_periods = self.attr['num_periods']

        edu_start = self.attr['edu_start']

        edu_max = self.attr['edu_max']

        # Extract auxiliary information
        shocks = model_paras['shocks']

        # Update derived attributes
        self.attr['min_idx'] = min(num_periods, (edu_max - edu_start + 1))

        self.attr['shocks_zero'] = (np.count_nonzero(shocks) == 0)

        self.attr['is_ambiguous'] = (self.attr['level'] > 0.00)

        self.attr['is_python'] = (self.attr['version'] == 'PYTHON')

    def _check_integrity_attributes(self):
        """ Check integrity of class instance. This testing is done the first
        time the class is locked and if the package is running in debug mode.
        """
        # Distribute class attributes
        is_interpolated = self.attr['is_interpolated']

        num_draws_emax = self.attr['num_draws_emax']

        num_draws_prob = self.attr['num_draws_prob']

        is_ambiguous = self.attr['is_ambiguous']

        num_periods = self.attr['num_periods']

        model_paras = self.attr['model_paras']

        num_agents = self.attr['num_agents']

        num_points = self.attr['num_points']

        edu_start = self.attr['edu_start']

        is_python = self.attr['is_python']

        seed_data = self.attr['seed_data']

        seed_prob = self.attr['seed_prob']

        seed_emax = self.attr['seed_emax']

        is_debug = self.attr['is_debug']

        measure = self.attr['measure']

        edu_max = self.attr['edu_max']

        version = self.attr['version']

        delta = self.attr['delta']

        level = self.attr['level']

        # Auxiliary objects
        shocks = model_paras['shocks']

        # Debug status
        assert (is_debug in [True, False])

        # Ambiguity in environment
        assert (is_ambiguous in [True, False])

        # Version of implementation
        assert (is_python in [True, False])

        # Constraints
        if is_ambiguous and version in ['F2PY', 'FORTRAN']:
            assert (measure in ['kl'])

        # Seeds
        for seed in [seed_emax, seed_data, seed_prob]:
            assert (np.isfinite(seed))
            assert (isinstance(seed, int))
            assert (seed > 0)

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
        assert (np.isfinite(num_draws_emax))
        assert (isinstance(num_draws_emax, int))
        assert (num_draws_emax >= 0)

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

        # Interpolation
        assert (is_interpolated in [True, False])
        assert (isinstance(num_points, int))
        assert (num_points > 0)

        # Simulation of S-ML
        assert (isinstance(num_draws_prob, int))
        assert (num_draws_prob > 0)

    def _check_integrity_results(self):
        """ This methods check the integrity of the results.
        """
        # Distribute auxiliary objects
        is_interpolated = self.attr['is_interpolated']

        num_periods = self.attr['num_periods']

        edu_start = self.attr['edu_start']

        edu_max = self.attr['edu_max']

        # Distribute results
        periods_payoffs_systematic = self.attr['periods_payoffs_systematic']

        periods_payoffs_future = self.attr['periods_payoffs_future']

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
                            period, index[0], index[1], index[2], index[3]]))
                    # Record finite value
                    is_infinite[
                        period, index[0], index[1], index[2], index[3]] = True
            # Check that all admissible states are finite
            assert (np.all(np.isfinite(mapping_state_idx[is_infinite == True])))

            # Check that all inadmissible states are infinite
            assert (
                np.all(np.isfinite(mapping_state_idx[is_infinite == False])) == False)

        # Check the calculated systematic payoffs
        is_applicable = (states_all is not None)
        is_applicable = is_applicable and (states_number_period is not None)
        is_applicable = is_applicable and (periods_payoffs_systematic is not None)

        if is_applicable:
            # Check that the payoffs are finite for all admissible values and
            # infinite for all others.
            is_infinite = np.tile(False, reps=periods_payoffs_systematic.shape)
            for period in range(num_periods):
                # Loop over all possible states
                for k in range(states_number_period[period]):
                    # Check that wages are all positive
                    assert (np.all(periods_payoffs_systematic[period, k, :2] > 0.0))
                    # Check for finite value at admissible state
                    assert (
                        np.all(np.isfinite(periods_payoffs_systematic[period, k, :])))
                    # Record finite value
                    is_infinite[period, k, :] = True
                # Check that all admissible states are finite
                assert (
                    np.all(np.isfinite(periods_payoffs_systematic[is_infinite ==
                                                               True])))
                # Check that all inadmissible states are infinite
                if num_periods > 1:
                    assert (np.all(np.isfinite(
                        periods_payoffs_systematic[is_infinite == False])) == False)

        # Check the expected future value
        is_applicable = (periods_emax is not None)
        is_applicable = is_applicable and (periods_payoffs_future is not None)

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
            # infinite for all others. This is only a valid request if no
            # interpolation is performed.
            if not is_interpolated:
                for period in range(num_periods - 1):
                    # Loop over all possible states
                    for k in range(states_number_period[period]):
                        # Check for finite value at admissible state, infinite
                        # values are allowed for the third column when the
                        # maximum level of education is attained.
                        assert (np.all(np.isfinite(periods_payoffs_future[period, k, :2])))
                        assert (np.all(np.isfinite(periods_payoffs_future[period, k, 3])))
                        # Special checks for infinite value due to
                        # high education.
                        if not np.isfinite(periods_payoffs_future[period, k, 2]):
                            assert (states_all[period, k][2] == edu_max - edu_start)

    def _check_key(self, key):
        """ Check that key is present.
        """
        # Check presence
        assert (key in self.attr.keys())

        # Finishing.
        return True
