""" Module that contains the class that carries around information for the
    RESPY package
"""

# standard library
import pickle as pkl
import pandas as pd
import numpy as np

# project library
from respy.python.shared.shared_auxiliary import replace_missing_values
from respy.python.shared.shared_auxiliary import check_model_parameters
from respy.python.estimate.estimate_auxiliary import dist_optim_paras
from respy.python.read.read_python import read

# Special care with derived attributes is required to maintain integrity of
# the class instance. These derived attributes cannot be changed directly.
DERIVED_ATTR = ['min_idx', 'is_myopic']

# Special care with solution attributes is required. These are only returned
# if the class instance was solved.
SOLUTION_ATTR = ['periods_payoffs_systematic', 'states_number_period']
SOLUTION_ATTR += ['mapping_state_idx', 'periods_emax', 'states_all']


class RespyCls(object):
    """ This class manages the distribution of the use requests throughout
    the toolbox.
    """

    def __init__(self, fname):
        """ Initialization of hand-crafted class for package management.
        """
        # Distribute class attributes
        self.attr = dict()

        self.attr['init_dict'] = read(fname)

        # Constitutive attributes
        self.attr['num_points_interp'] = None

        self.attr['optimizer_options'] = None

        self.attr['is_interpolated'] = None

        self.attr['num_draws_emax'] = None

        self.attr['num_draws_prob'] = None

        self.attr['optimizer_used'] = None

        self.attr['num_agents_sim'] = None

        self.attr['num_agents_est'] = None

        self.attr['paras_fixed'] = None

        self.attr['num_periods'] = None

        self.attr['model_paras'] = None

        self.attr['is_parallel'] = None

        self.attr['num_procs'] = None

        self.attr['seed_prob'] = None

        self.attr['seed_emax'] = None

        self.attr['is_locked'] = None

        self.attr['is_solved'] = None

        self.attr['edu_start'] = None

        self.attr['seed_sim'] = None

        self.attr['is_debug'] = None

        self.attr['file_sim'] = None

        self.attr['file_est'] = None

        self.attr['edu_max'] = None

        self.attr['version'] = None

        self.attr['maxfun'] = None

        self.attr['delta'] = None

        self.attr['store'] = None

        self.attr['tau'] = None

        # Derived attributes
        self.attr['is_myopic'] = None

        self.attr['min_idx'] = None

        # Solution attributes
        self.attr['periods_payoffs_systematic'] = None

        self.attr['states_number_period'] = None

        self.attr['mapping_state_idx'] = None

        self.attr['periods_emax'] = None

        self.attr['states_all'] = None

        # Initialization
        self._update_core_attributes()

        self._update_derived_attributes()

        # Status indicators
        self.attr['is_locked'] = False

        self.attr['is_solved'] = False

        self.lock()

    def update_model_paras(self, x):
        """ Update model parameters.
        """
        # Determine use of interface
        coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky = \
                    dist_optim_paras(x, True)

        # Check integrity
        check_model_parameters(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
            shocks_cholesky)

        # Distribute class attributes
        model_paras = self.attr['model_paras']

        # Update model parametrization
        model_paras['coeffs_home'] = coeffs_home

        model_paras['coeffs_edu'] = coeffs_edu

        model_paras['coeffs_a'] = coeffs_a

        model_paras['coeffs_b'] = coeffs_b

        model_paras['shocks_cholesky'] = shocks_cholesky

        # Update class attributes
        self.attr['model_paras'] = model_paras

    def lock(self):
        """ Lock class instance.
        """
        # Antibugging.
        assert (not self.attr['is_locked'])

        # Checks
        self._check_integrity_attributes()

        self._check_integrity_results()

        # Update status indicator
        self.attr['is_locked'] = True

    def unlock(self):
        """ Unlock class instance.
        """
        # Antibugging
        assert self.attr['is_locked']

        # Update status indicator
        self.attr['is_locked'] = False

    def get_attr(self, key):
        """ Get attributes.
        """
        # Antibugging
        assert self.attr['is_locked']
        assert self._check_key(key)

        # If solution attributes are requested, make sure the class instance
        # is solved.
        if key in SOLUTION_ATTR:
            assert self.get_attr('is_solved'), 'invalid request'

        # Finishing
        return self.attr[key]

    def set_attr(self, key, value):
        """ Get attributes.
        """
        # Antibugging
        assert (not self.attr['is_locked'])
        assert self._check_key(key)

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

        # Special care is required for solution attributes. These cannot be
        # set when the class instance is solved. The status attribute is
        # accessed directly as the class instance is unlocked, which does not
        # allow to access attributes using the get method.
        if key in SOLUTION_ATTR:
            assert not self.attr['is_solved'], 'invalid request'

        # Update derived attributes
        self._update_derived_attributes()

    def store(self, file_name):
        """ Store class instance.
        """
        # Antibugging
        assert self.attr['is_locked']
        assert isinstance(file_name, str)

        # Store.
        pkl.dump(self, open(file_name, 'wb'))

    def reset(self):
        """ Remove solution attributes from class instance.
        """
        for label in SOLUTION_ATTR:
            self.attr[label] = None

        self.attr['is_solved'] = False

    def check_equal_solution(self, other):
        """ This method allows to compare two class instances with respect to the equality of their solution attributes.
        """
        assert (isinstance(other, RespyCls))

        for key_ in SOLUTION_ATTR:
            try:
                np.testing.assert_almost_equal(self.attr[key_], other.attr[key_])
            except AssertionError:
                return False

        return True

    def _update_core_attributes(self):
        """ Calculate derived attributes. This is only called when the class
        is initialized
        """
        # Distribute class attributes
        init_dict = self.attr['init_dict']

        # Extract information from initialization dictionary and construct
        # auxiliary objects.
        self.attr['is_interpolated'] = init_dict['INTERPOLATION']['flag']

        self.attr['optimizer_used'] = init_dict['ESTIMATION']['optimizer']

        self.attr['num_agents_sim'] = init_dict['SIMULATION']['agents']

        self.attr['num_agents_est'] = init_dict['ESTIMATION']['agents']

        self.attr['num_draws_prob'] = init_dict['ESTIMATION']['draws']

        self.attr['is_parallel'] = init_dict['PARALLELISM']['flag']

        self.attr['num_points_interp'] = init_dict['INTERPOLATION']['points']

        self.attr['num_draws_emax'] = init_dict['SOLUTION']['draws']

        self.attr['num_procs'] = init_dict['PARALLELISM']['procs']

        self.attr['num_periods'] = init_dict['BASICS']['periods']

        self.attr['maxfun'] = init_dict['ESTIMATION']['maxfun']

        self.attr['edu_start'] = init_dict['EDUCATION']['start']

        self.attr['seed_sim'] = init_dict['SIMULATION']['seed']

        self.attr['seed_prob'] = init_dict['ESTIMATION']['seed']

        self.attr['file_sim'] = init_dict['SIMULATION']['file']

        self.attr['file_est'] = init_dict['ESTIMATION']['file']

        self.attr['seed_emax'] = init_dict['SOLUTION']['seed']

        self.attr['version'] = init_dict['PROGRAM']['version']

        self.attr['is_debug'] = init_dict['PROGRAM']['debug']

        self.attr['edu_max'] = init_dict['EDUCATION']['max']

        self.attr['store'] = init_dict['SOLUTION']['store']

        self.attr['delta'] = init_dict['BASICS']['delta']

        self.attr['tau'] = init_dict['ESTIMATION']['tau']

        # Initialize model parameters
        self.attr['model_paras'] = dict()

        # Constructing the covariance matrix of the shocks
        shocks_coeffs = init_dict['SHOCKS']['coeffs']
        for i in [0, 4, 7, 9]:
            shocks_coeffs[i] **= 2

        shocks = np.zeros((4, 4))
        shocks[0, :] = shocks_coeffs[0:4]
        shocks[1, 1:] = shocks_coeffs[4:7]
        shocks[2, 2:] = shocks_coeffs[7:9]
        shocks[3, 3:] = shocks_coeffs[9:10]

        shocks_cov = shocks + shocks.T - np.diag(shocks.diagonal())

        # As we call the Cholesky decomposition, we need to handle the
        # special case of a deterministic model.
        if np.count_nonzero(shocks_cov) == 0:
            self.attr['model_paras']['shocks_cholesky'] = np.zeros((4, 4))
        else:
            shocks_cholesky = np.linalg.cholesky(shocks_cov)
            self.attr['model_paras']['shocks_cholesky'] = shocks_cholesky

        self.attr['model_paras']['coeffs_a'] = \
            init_dict['OCCUPATION A']['coeffs']
        self.attr['model_paras']['coeffs_b'] = \
            init_dict['OCCUPATION B']['coeffs']
        self.attr['model_paras']['coeffs_edu'] = \
            init_dict['EDUCATION']['coeffs']
        self.attr['model_paras']['coeffs_home'] = \
            init_dict['HOME']['coeffs']

        # Initialize information about optimization parameters
        self.attr['paras_fixed'] = init_dict['OCCUPATION A']['fixed'][:]
        self.attr['paras_fixed'] += init_dict['OCCUPATION B']['fixed'][:]
        self.attr['paras_fixed'] += init_dict['EDUCATION']['fixed'][:]
        self.attr['paras_fixed'] += init_dict['HOME']['fixed'][:]
        self.attr['paras_fixed'] += init_dict['SHOCKS']['fixed'].tolist()

        # Ensure that all elements in the dictionary are of the same
        # type.
        keys = ['coeffs_a', 'coeffs_b', 'coeffs_edu', 'coeffs_home']
        keys += ['shocks_cholesky']
        for key_ in keys:
            self.attr['model_paras'][key_] = \
                np.array(self.attr['model_paras'][key_])

        # Aggregate all the information provided about optimizer options in
        # one class attribute for easier access later.
        optimizers = ['SCIPY-BFGS', 'SCIPY-POWELL', 'FORT-NEWUOA', 'FORT-BFGS']
        self.attr['optimizer_options'] = dict()
        for optimizer in optimizers:
            is_defined = (optimizer in init_dict.keys())
            if is_defined:
                self.attr['optimizer_options'][optimizer] = \
                    init_dict[optimizer]

        # Delete the duplicated information from the initialization
        # dictionary. Special treatment of EDUCATION is required as it
        # contains other information about education than just the
        # payoff parametrization.
        del self.attr['init_dict']

    def _update_derived_attributes(self):
        """ Update derived attributes.
        """
        # Distribute model parameters
        num_periods = self.attr['num_periods']

        edu_start = self.attr['edu_start']

        edu_max = self.attr['edu_max']

        # Update derived attributes
        self.attr['min_idx'] = min(num_periods, (edu_max - edu_start + 1))

        self.attr['is_myopic'] = (self.attr['delta'] == 0.00)

    def _check_integrity_attributes(self):
        """ Check integrity of class instance. This testing is done the first
        time the class is locked and if the package is running in debug mode.
        """
        # Distribute class attributes
        is_interpolated = self.attr['is_interpolated']

        optimizer_used = self.attr['optimizer_used']

        num_draws_emax = self.attr['num_draws_emax']

        num_draws_prob = self.attr['num_draws_prob']

        num_agents_sim = self.attr['num_agents_sim']

        num_agents_est = self.attr['num_agents_est']

        is_parallel = self.attr['is_parallel']

        paras_fixed = self.attr['paras_fixed']

        num_periods = self.attr['num_periods']

        model_paras = self.attr['model_paras']

        num_points_interp = self.attr['num_points_interp']

        edu_start = self.attr['edu_start']

        is_myopic = self.attr['is_myopic']

        seed_sim = self.attr['seed_sim']

        seed_prob = self.attr['seed_prob']

        seed_emax = self.attr['seed_emax']

        num_procs = self.attr['num_procs']

        is_debug = self.attr['is_debug']

        edu_max = self.attr['edu_max']

        version = self.attr['version']

        maxfun = self.attr['maxfun']

        delta = self.attr['delta']

        tau = self.attr['tau']

        # Auxiliary objects
        shocks_cholesky = model_paras['shocks_cholesky']

        # Parallelism
        assert (is_parallel in [True, False])
        assert (num_procs > 0)
        if is_parallel:
            assert (version == 'FORTRAN')

        # Status of optimization parameters
        assert isinstance(paras_fixed, list)
        assert (len(paras_fixed) == 26)
        assert (np.all(paras_fixed) in [True, False])

        # Debug status
        assert (is_debug in [True, False])

        # Forward-looking agents
        assert (is_myopic in [True, False])

        # Seeds
        for seed in [seed_emax, seed_sim, seed_prob]:
            assert (np.isfinite(seed))
            assert (isinstance(seed, int))
            assert (seed > 0)

        # Number of agents
        for num_agents in [num_agents_sim, num_agents_est]:
            assert (np.isfinite(num_agents))
            assert (isinstance(num_agents, int))
            assert (num_agents > 0)

        # Number of periods
        assert (np.isfinite(num_periods))
        assert (isinstance(num_periods, int))
        assert (num_periods > 0)

        # Start of education level
        assert (np.isfinite(edu_start))
        assert (isinstance(edu_start, int))
        assert (edu_start >= 0)

        # Number of draws for Monte Carlo integration
        assert (np.isfinite(num_draws_emax))
        assert (isinstance(num_draws_emax, int))
        assert (num_draws_emax >= 0)

        # Maximum level of education
        assert (np.isfinite(edu_max))
        assert (isinstance(edu_max, int))
        assert (edu_max >= 0)
        assert (edu_max >= edu_start)

        # Debugging mode
        assert (is_debug in [True, False])

        # Window for smoothing parameter
        assert (isinstance(tau, float))
        assert (tau > 0)

        # Discount factor
        assert (np.isfinite(delta))
        assert (isinstance(delta, float))
        assert (delta >= 0.00)

        # Version version of package
        assert (version in ['FORTRAN', 'PYTHON'])

        # Shock distribution
        assert (isinstance(shocks_cholesky, np.ndarray))
        assert (np.all(np.isfinite(shocks_cholesky)))
        assert (shocks_cholesky.shape == (4, 4))

        # Interpolation
        assert (is_interpolated in [True, False])
        assert (isinstance(num_points_interp, int))
        assert (num_points_interp > 0)

        # Simulation of S-ML
        assert (isinstance(num_draws_prob, int))
        assert (num_draws_prob > 0)

        # Maximum number of iterations
        assert (isinstance(maxfun, int))
        assert (maxfun >= 0)

        # Optimizers
        assert (optimizer_used in ['SCIPY-BFGS', 'SCIPY-POWELL',
            'FORT-NEWUOA', 'FORT-BFGS'])

    def _check_integrity_results(self):
        """ This methods check the integrity of the results.
        """
        # Check if solution attributes well maintained.
        for label in SOLUTION_ATTR:
            if self.attr['is_solved']:
                assert (self.attr[label] is not None)
            else:
                assert (self.attr[label] is None)

        # Distribute class attributes
        num_periods = self.attr['num_periods']

        edu_start = self.attr['edu_start']

        edu_max = self.attr['edu_max']

        # Distribute results
        periods_payoffs_systematic = self.attr['periods_payoffs_systematic']

        states_number_period = self.attr['states_number_period']

        mapping_state_idx = self.attr['mapping_state_idx']

        periods_emax = self.attr['periods_emax']

        states_all = self.attr['states_all']

        # Replace missing value with NAN. This allows to easily select the
        # valid subsets of the containers
        if mapping_state_idx is not None:
            mapping_state_idx = replace_missing_values(mapping_state_idx)
        if states_all is not None:
            states_all = replace_missing_values(states_all)
        if periods_payoffs_systematic is not None:
            periods_payoffs_systematic = replace_missing_values(periods_payoffs_systematic)
        if periods_emax is not None:
            periods_emax = replace_missing_values(periods_emax)

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
                assert (np.all(
                    states_all[period, :states_number_period[period]] >= 0))

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
                assert (np.all(np.isfinite(
                    states_all[period, :states_number_period[period]])))

            # There are no infinite values in final period.
            assert (np.all(np.isfinite(states_all[(num_periods - 1), :, :])))

            # There are is only one finite realization in period one.
            assert (np.sum(np.isfinite(mapping_state_idx[0, :, :, :, :])) == 1)

            # If valid, the number of state space realizations in period two is
            # four.
            if num_periods > 1:
                assert (
                np.sum(np.isfinite(mapping_state_idx[1, :, :, :, :])) == 4)

            # Check that mapping is defined for all possible realizations of the
            # state space by period. Check that mapping is not defined for all
            # inadmissible values.
            is_infinite = np.tile(False, reps=mapping_state_idx.shape)
            for period in range(num_periods):
                # Subsetting valid indices
                indices = states_all[period, :states_number_period[
                    period], :].astype('int')
                for index in indices:
                    # Check for finite value at admissible state
                    assert (np.isfinite(mapping_state_idx[period, index[0],
                        index[1], index[2], index[3]]))
                    # Record finite value
                    is_infinite[
                        period, index[0], index[1], index[2], index[3]] = True
            # Check that all admissible states are finite
            assert (np.all(np.isfinite(mapping_state_idx[is_infinite == True])))

            # Check that all inadmissible states are infinite
            assert (np.all(np.isfinite(
                mapping_state_idx[is_infinite == False])) == False)

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
                    assert (np.all(periods_payoffs_systematic[period, k, :2] >= 0.0))
                    # Check for finite value at admissible state
                    assert (np.all(np.isfinite(periods_payoffs_systematic[
                    period, k, :])))
                    # Record finite value
                    is_infinite[period, k, :] = True
                # Check that all admissible states are finite
                assert (np.all(np.isfinite(periods_payoffs_systematic[
                    is_infinite == True])))
                # Check that all inadmissible states are infinite
                if num_periods > 1:
                    assert (np.all(np.isfinite(periods_payoffs_systematic[is_infinite == False])) == False)

        # Check the expected future value
        is_applicable = (periods_emax is not None)

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
                    assert (np.all(np.isfinite(periods_emax[is_infinite == False])) == False)

    def _check_key(self, key):
        """ Check that key is present.
        """
        # Check presence
        assert (key in self.attr.keys())

        # Finishing.
        return True

