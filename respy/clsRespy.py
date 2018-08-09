import pickle as pkl
import pandas as pd
import numpy as np
import json
import copy

from respy.python.shared.shared_auxiliary import replace_missing_values
from respy.python.shared.shared_auxiliary import check_model_parameters
from respy.python.shared.shared_auxiliary import cholesky_to_coeffs
from respy.python.shared.shared_auxiliary import print_init_dict
from respy.python.shared.shared_auxiliary import dist_econ_paras
from respy.python.shared.shared_auxiliary import get_optim_paras
from respy.python.shared.shared_constants import OPT_EST_FORT
from respy.python.shared.shared_constants import OPT_EST_PYTH
from respy.python.shared.shared_constants import PRINT_FLOAT
from respy.python.shared.shared_constants import IS_PARALLEL
from respy.python.shared.shared_constants import IS_FORTRAN
from respy.python.shared.shared_constants import ROOT_DIR
from respy.python.read.read_python import read
from respy.custom_exceptions import UserError


# Define classes of attributes that need special treatment

# Derived attributes cannot be set by the user
DERIVED_ATTR = ['is_myopic']

# Solution attributes are only defined if the class instance was solved.
SOLUTION_ATTR = ['periods_rewards_systematic', 'states_number_period',
                 'mapping_state_idx', 'periods_emax', 'states_all']

# Define list of admissible optimizers
OPTIMIZERS = OPT_EST_FORT + OPT_EST_PYTH

# We need to do some reorganization as the parameters from the initialization
# that describe the covariance of the shocks need to be mapped to the
# Cholesky factors which are the parameters the optimizer actually iterates on.
PARAS_MAPPING = [(43, 43), (44, 44), (45, 46), (46, 49), (47, 45), (48, 47),
                 (49, 50), (50, 48), (51, 51), (52, 52)]


class RespyCls(object):
    """Class that defines a model in respy.  """

    def __init__(self, fname):
        """ Initialization of hand-crafted class for package management.
        """
        # Distribute class attributes
        self.attr = {}
        self.attr['init_dict'] = read(fname)
        self._initialize_attributes()
        self._update_core_attributes()
        self._update_derived_attributes()

        # Status indicators
        self.attr['is_locked'] = False
        self.attr['is_solved'] = False
        self.lock()

    def _initialize_attributes(self):
        """Initialize self.attr.

        self.attr contains the constitutive attribute that define a model in
        respy. Here most of them are initialized to None, later they will be
        updated from the initialization file.

        """
        initialize_to_none = [
            'num_points_interp', 'optimizer_options', 'is_interpolated',
            'num_draws_emax', 'num_draws_prob', 'optimizer_used',
            'num_agents_sim', 'num_agents_est', 'num_periods', 'optim_paras',
            'derivatives', 'num_procs', 'seed_prob', 'num_types', 'seed_emax',
            'is_locked', 'is_solved', 'seed_sim', 'is_debug', 'file_sim',
            'file_est', 'is_store', 'version', 'maxfun', 'tau', 'is_myopic',
            'periods_rewards_systematic', 'states_number_period',
            'mapping_state_idx', 'periods_emax', 'states_all']

        for attribute in initialize_to_none:
            self.attr[attribute] = None

        precond_spec = {'minimum': None, 'type': None, 'eps': None}
        self.attr['precond_spec'] = precond_spec

        edu_spec = {'lagged': None, 'start': None, 'share': None, 'max': None}
        self.attr['educ_spec'] = edu_spec

    def update_optim_paras(self, x_econ):
        """ Update model parameters.
        """
        x_econ = copy.deepcopy(x_econ)

        self.reset()

        # Determine use of interface
        delta, coeffs_common, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, \
            shocks_cov, type_shares, type_shifts = dist_econ_paras(x_econ)

        shocks_cholesky = np.linalg.cholesky(shocks_cov)

        # Distribute class attributes
        optim_paras = self.attr['optim_paras']

        # Update model parametrization
        optim_paras['shocks_cholesky'] = shocks_cholesky

        optim_paras['coeffs_common'] = coeffs_common

        optim_paras['coeffs_home'] = coeffs_home

        optim_paras['coeffs_edu'] = coeffs_edu

        optim_paras['coeffs_a'] = coeffs_a

        optim_paras['coeffs_b'] = coeffs_b

        optim_paras['delta'] = delta

        optim_paras['type_shares'] = type_shares

        optim_paras['type_shifts'] = type_shifts

        # Check integrity
        check_model_parameters(optim_paras)

        # Update class attributes
        self.attr['optim_paras'] = optim_paras

    def lock(self):
        """ Lock class instance."""
        assert (not self.attr['is_locked']), \
            'Only unlocked instances of clsRespy can be locked.'

        self._update_derived_attributes()
        self._check_integrity_attributes()
        self._check_integrity_results()
        self.attr['is_locked'] = True

    def unlock(self):
        """ Unlock class instance."""
        assert self.attr['is_locked'], \
            'Only locked instances of clsRespy can be unlocked.'

        self.attr['is_locked'] = False

    def get_attr(self, key):
        """Get attributes."""
        assert self.attr['is_locked']
        self._check_key(key)

        if key in SOLUTION_ATTR:
            assert self.get_attr('is_solved'), 'invalid request'

        return self.attr[key]

    def set_attr(self, key, value):
        """Set attributes."""
        assert (not self.attr['is_locked'])
        self._check_key(key)

        invalid_attr = DERIVED_ATTR + ['optim_paras', 'init_dict']
        if key in invalid_attr:
            raise AssertionError(
                '{} must not be modified by users!'.format(key))

        if key in SOLUTION_ATTR:
            assert not self.attr['is_solved'], \
                'Solution attributes can only be set if model is not solved.'

        self.attr[key] = value
        self._update_derived_attributes()

    def store(self, file_name):
        """Store class instance."""
        assert self.attr['is_locked']
        assert isinstance(file_name, str)
        pkl.dump(self, open(file_name, 'wb'))

    def write_out(self, fname='model.respy.ini'):
        """Write out the implied initialization file of the class instance."""
        # Distribute class attributes
        num_paras = self.attr['num_paras']

        num_types = self.attr['num_types']

        # We reconstruct the initialization dictionary as otherwise we need
        # to constantly update the original one.
        init_dict = dict()

        # Basics
        init_dict['BASICS'] = dict()
        init_dict['BASICS']['periods'] = self.attr['num_periods']
        init_dict['BASICS']['coeffs'] = self.attr['optim_paras']['delta']
        init_dict['BASICS']['bounds'] = \
            self.attr['optim_paras']['paras_bounds'][0:1]
        init_dict['BASICS']['fixed'] = \
            self.attr['optim_paras']['paras_fixed'][0:1]

        # Common Returns
        lower, upper = 1, 3
        init_dict['COMMON'] = dict()
        init_dict['COMMON']['coeffs'] = \
            self.attr['optim_paras']['coeffs_common']
        init_dict['COMMON']['bounds'] = \
            self.attr['optim_paras']['paras_bounds'][lower:upper]
        init_dict['COMMON']['fixed'] = \
            self.attr['optim_paras']['paras_fixed'][lower:upper]

        # Occupation A
        lower, upper = 3, 18
        init_dict['OCCUPATION A'] = dict()
        init_dict['OCCUPATION A']['coeffs'] = \
            self.attr['optim_paras']['coeffs_a']

        init_dict['OCCUPATION A']['bounds'] = \
            self.attr['optim_paras']['paras_bounds'][lower:upper]
        init_dict['OCCUPATION A']['fixed'] = \
            self.attr['optim_paras']['paras_fixed'][lower:upper]

        # Occupation B
        lower, upper = 18, 33
        init_dict['OCCUPATION B'] = dict()
        init_dict['OCCUPATION B']['coeffs'] = \
            self.attr['optim_paras']['coeffs_b']

        init_dict['OCCUPATION B']['bounds'] = \
            self.attr['optim_paras']['paras_bounds'][lower:upper]
        init_dict['OCCUPATION B']['fixed'] = \
            self.attr['optim_paras']['paras_fixed'][lower:upper]

        # Education
        lower, upper = 33, 40
        init_dict['EDUCATION'] = dict()
        init_dict['EDUCATION']['coeffs'] = \
            self.attr['optim_paras']['coeffs_edu']

        init_dict['EDUCATION']['bounds'] = \
            self.attr['optim_paras']['paras_bounds'][lower:upper]
        init_dict['EDUCATION']['fixed'] = \
            self.attr['optim_paras']['paras_fixed'][lower:upper]

        init_dict['EDUCATION']['lagged'] = self.attr['edu_spec']['lagged']
        init_dict['EDUCATION']['start'] = self.attr['edu_spec']['start']
        init_dict['EDUCATION']['share'] = self.attr['edu_spec']['share']
        init_dict['EDUCATION']['max'] = self.attr['edu_spec']['max']

        # Home
        lower, upper = 40, 43
        init_dict['HOME'] = dict()
        init_dict['HOME']['coeffs'] = self.attr['optim_paras']['coeffs_home']

        init_dict['HOME']['bounds'] = \
            self.attr['optim_paras']['paras_bounds'][lower:upper]
        init_dict['HOME']['fixed'] = \
            self.attr['optim_paras']['paras_fixed'][lower:upper]

        # Shocks
        lower, upper = 43, 53
        init_dict['SHOCKS'] = dict()
        shocks_cholesky = self.attr['optim_paras']['shocks_cholesky']
        shocks_coeffs = cholesky_to_coeffs(shocks_cholesky)
        init_dict['SHOCKS']['coeffs'] = shocks_coeffs

        init_dict['SHOCKS']['bounds'] = \
            self.attr['optim_paras']['paras_bounds'][lower:upper]

        # Again we need to reorganize the order of the coefficients
        paras_fixed_reordered = self.attr['optim_paras']['paras_fixed'][:]

        paras_fixed = paras_fixed_reordered[:]
        for old, new in PARAS_MAPPING:
            paras_fixed[old] = paras_fixed_reordered[new]

        init_dict['SHOCKS']['fixed'] = paras_fixed[43:53]

        # Solution
        init_dict['SOLUTION'] = dict()
        init_dict['SOLUTION']['draws'] = self.attr['num_draws_emax']
        init_dict['SOLUTION']['seed'] = self.attr['seed_emax']
        init_dict['SOLUTION']['store'] = self.attr['is_store']

        # Type Shares
        lower, upper = 53, 53 + (num_types - 1) * 2
        init_dict['TYPE SHARES'] = dict()
        init_dict['TYPE SHARES']['coeffs'] = \
            self.attr['optim_paras']['type_shares'][2:]
        init_dict['TYPE SHARES']['bounds'] = \
            self.attr['optim_paras']['paras_bounds'][lower:upper]
        init_dict['TYPE SHARES']['fixed'] = \
            self.attr['optim_paras']['paras_fixed'][lower:upper]

        # Type Shifts
        lower, upper = 53 + (num_types - 1) * 2, num_paras
        init_dict['TYPE SHIFTS'] = dict()
        init_dict['TYPE SHIFTS']['coeffs'] = \
            self.attr['optim_paras']['type_shifts'].flatten()[4:]
        init_dict['TYPE SHIFTS']['bounds'] = \
            self.attr['optim_paras']['paras_bounds'][lower:upper]
        init_dict['TYPE SHIFTS']['fixed'] = \
            self.attr['optim_paras']['paras_fixed'][lower:upper]

        # Simulation
        init_dict['SIMULATION'] = dict()
        init_dict['SIMULATION']['agents'] = self.attr['num_agents_sim']
        init_dict['SIMULATION']['file'] = self.attr['file_sim']
        init_dict['SIMULATION']['seed'] = self.attr['seed_sim']

        # Estimation
        init_dict['ESTIMATION'] = dict()
        init_dict['ESTIMATION']['optimizer'] = self.attr['optimizer_used']
        init_dict['ESTIMATION']['agents'] = self.attr['num_agents_est']
        init_dict['ESTIMATION']['draws'] = self.attr['num_draws_prob']
        init_dict['ESTIMATION']['seed'] = self.attr['seed_prob']
        init_dict['ESTIMATION']['file'] = self.attr['file_est']
        init_dict['ESTIMATION']['maxfun'] = self.attr['maxfun']
        init_dict['ESTIMATION']['tau'] = self.attr['tau']

        # Derivatives
        init_dict['DERIVATIVES'] = dict()
        init_dict['DERIVATIVES']['version'] = self.attr['derivatives']

        # Scaling
        init_dict['PRECONDITIONING'] = dict()
        init_dict['PRECONDITIONING']['minimum'] = \
            self.attr['precond_spec']['minimum']
        init_dict['PRECONDITIONING']['type'] = \
            self.attr['precond_spec']['type']
        init_dict['PRECONDITIONING']['eps'] = self.attr['precond_spec']['eps']

        # Program
        init_dict['PROGRAM'] = dict()
        init_dict['PROGRAM']['version'] = self.attr['version']
        init_dict['PROGRAM']['procs'] = self.attr['num_procs']
        init_dict['PROGRAM']['debug'] = self.attr['is_debug']

        # Interpolation
        init_dict['INTERPOLATION'] = dict()
        init_dict['INTERPOLATION']['points'] = self.attr['num_points_interp']
        init_dict['INTERPOLATION']['flag'] = self.attr['is_interpolated']

        # Optimizers
        for optimizer in self.attr['optimizer_options'].keys():
            init_dict[optimizer] = self.attr['optimizer_options'][optimizer]

        print_init_dict(init_dict, fname)

    def reset(self):
        """ Remove solution attributes from class instance.
        """
        for label in SOLUTION_ATTR:
            self.attr[label] = None

        self.attr['is_solved'] = False

    def check_equal_solution(self, other):
        """ Compare two class instances for equality of solution attributes."""
        assert (isinstance(other, RespyCls))

        for key_ in SOLUTION_ATTR:
            try:
                np.testing.assert_almost_equal(
                    self.attr[key_], other.attr[key_])
            except AssertionError:
                return False

        return True

    def _update_core_attributes(self):
        """Only called when the class is initialized."""
        # Distribute class attributes
        init_dict = self.attr['init_dict']

        # Extract information from initialization dictionary and construct
        # auxiliary objects.
        self.attr['num_points_interp'] = init_dict['INTERPOLATION']['points']

        self.attr['optimizer_used'] = init_dict['ESTIMATION']['optimizer']

        self.attr['is_interpolated'] = init_dict['INTERPOLATION']['flag']

        self.attr['num_agents_sim'] = init_dict['SIMULATION']['agents']

        self.attr['num_agents_est'] = init_dict['ESTIMATION']['agents']

        self.attr['derivatives'] = init_dict['DERIVATIVES']['version']

        self.attr['num_draws_prob'] = init_dict['ESTIMATION']['draws']

        self.attr['num_draws_emax'] = init_dict['SOLUTION']['draws']

        self.attr['num_periods'] = init_dict['BASICS']['periods']

        self.attr['seed_prob'] = init_dict['ESTIMATION']['seed']

        self.attr['maxfun'] = init_dict['ESTIMATION']['maxfun']

        self.attr['seed_sim'] = init_dict['SIMULATION']['seed']

        self.attr['file_sim'] = init_dict['SIMULATION']['file']

        self.attr['file_est'] = init_dict['ESTIMATION']['file']

        self.attr['is_store'] = init_dict['SOLUTION']['store']

        self.attr['seed_emax'] = init_dict['SOLUTION']['seed']

        self.attr['version'] = init_dict['PROGRAM']['version']

        self.attr['num_procs'] = init_dict['PROGRAM']['procs']

        self.attr['is_debug'] = init_dict['PROGRAM']['debug']

        self.attr['edu_max'] = init_dict['EDUCATION']['max']

        self.attr['tau'] = init_dict['ESTIMATION']['tau']

        self.attr['precond_spec'] = dict()
        self.attr['precond_spec']['minimum'] = \
            init_dict['PRECONDITIONING']['minimum']
        self.attr['precond_spec']['type'] = \
            init_dict['PRECONDITIONING']['type']
        self.attr['precond_spec']['eps'] = init_dict['PRECONDITIONING']['eps']

        self.attr['edu_spec'] = dict()
        self.attr['edu_spec']['lagged'] = init_dict['EDUCATION']['lagged']
        self.attr['edu_spec']['start'] = init_dict['EDUCATION']['start']
        self.attr['edu_spec']['share'] = init_dict['EDUCATION']['share']
        self.attr['edu_spec']['max'] = init_dict['EDUCATION']['max']

        self.attr['num_types'] = \
            int(len(init_dict['TYPE SHARES']['coeffs']) / 2) + 1

        # Initialize model parameters
        self.attr['optim_paras'] = dict()

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
            self.attr['optim_paras']['shocks_cholesky'] = np.zeros((4, 4))
        else:
            shocks_cholesky = np.linalg.cholesky(shocks_cov)
            self.attr['optim_paras']['shocks_cholesky'] = shocks_cholesky

        # Constructing the shifts for each type.
        type_shifts = init_dict['TYPE SHIFTS']['coeffs']
        type_shares = init_dict['TYPE SHARES']['coeffs']

        if self.attr['num_types'] == 1:
            type_shares = np.tile(0.0, 2)
            type_shifts = np.tile(0.0, (1, 4))
        else:
            type_shares = np.concatenate(
                (np.tile(0.0, 2), type_shares), axis=0)
            type_shifts = np.reshape(
                type_shifts, (self.attr['num_types'] - 1, 4))
            type_shifts = np.concatenate(
                (np.tile(0.0, (1, 4)), type_shifts), axis=0)

        self.attr['optim_paras']['type_shifts'] = type_shifts
        self.attr['optim_paras']['type_shares'] = type_shares

        self.attr['optim_paras']['coeffs_a'] = \
            init_dict['OCCUPATION A']['coeffs']
        self.attr['optim_paras']['coeffs_b'] = \
            init_dict['OCCUPATION B']['coeffs']
        self.attr['optim_paras']['coeffs_common'] = \
            init_dict['COMMON']['coeffs']
        self.attr['optim_paras']['coeffs_edu'] = \
            init_dict['EDUCATION']['coeffs']
        self.attr['optim_paras']['coeffs_home'] = init_dict['HOME']['coeffs']
        self.attr['optim_paras']['delta'] = init_dict['BASICS']['coeffs']

        # Initialize information about optimization parameters
        keys = ['BASICS', 'COMMON', 'OCCUPATION A', 'OCCUPATION B',
                'EDUCATION', 'HOME', 'SHOCKS', 'TYPE SHARES', 'TYPE SHIFTS']

        for which in ['fixed', 'bounds']:
            self.attr['optim_paras']['paras_' + which] = []
            for key_ in keys:
                self.attr['optim_paras']['paras_' + which] += \
                    init_dict[key_][which][:]

        # Ensure that all elements in the dictionary are of the same type.
        keys = []
        keys += ['coeffs_a', 'coeffs_b', 'coeffs_edu', 'coeffs_home']
        keys += ['shocks_cholesky', 'delta', 'type_shares']
        keys += ['type_shifts', 'coeffs_common']
        for key_ in keys:
            self.attr['optim_paras'][key_] = \
                np.array(self.attr['optim_paras'][key_])

        # Aggregate all the information provided about optimizer options in
        # one class attribute for easier access later.
        self.attr['optimizer_options'] = dict()
        for optimizer in OPTIMIZERS:
            is_defined = (optimizer in init_dict.keys())
            if is_defined:
                self.attr['optimizer_options'][optimizer] = \
                    init_dict[optimizer]

        # We need to align the indicator for the fixed parameters.
        # In the initialization file, these refer to the upper triangular
        # matrix of the covariances. Inside the  program, we use the lower
        # triangular Cholesky decomposition.
        paras_fixed = self.attr['optim_paras']['paras_fixed'][:]

        paras_fixed_reordered = paras_fixed[:]
        for old, new in PARAS_MAPPING:
            paras_fixed_reordered[new] = paras_fixed[old]

        self.attr['optim_paras']['paras_fixed'] = paras_fixed_reordered

        # Delete the duplicated information from the initialization dictionary.
        # Special treatment of EDUCATION is required as it contains other
        # information about education than just the rewards parametrization.
        del self.attr['init_dict']

    def _update_derived_attributes(self):
        """Update derived attributes."""
        num_types = self.attr['num_types']
        self.attr['is_myopic'] = (self.attr['optim_paras']['delta'] == 0.00)[0]
        self.attr['num_paras'] = 53 + (num_types - 1) * 6

    def _check_integrity_attributes(self):
        """Check integrity of class instance.

        This testing is done the first time the class is locked and if
        the package is running in debug mode.

        """
        # Distribute class attributes
        num_points_interp = self.attr['num_points_interp']

        is_interpolated = self.attr['is_interpolated']

        optimizer_used = self.attr['optimizer_used']

        num_draws_emax = self.attr['num_draws_emax']

        num_draws_prob = self.attr['num_draws_prob']

        num_agents_sim = self.attr['num_agents_sim']

        num_agents_est = self.attr['num_agents_est']

        precond_spec = self.attr['precond_spec']

        derivatives = self.attr['derivatives']

        num_periods = self.attr['num_periods']

        optim_paras = self.attr['optim_paras']

        edu_spec = self.attr['edu_spec']

        is_myopic = self.attr['is_myopic']

        seed_prob = self.attr['seed_prob']

        seed_emax = self.attr['seed_emax']

        num_procs = self.attr['num_procs']

        num_paras = self.attr['num_paras']

        is_debug = self.attr['is_debug']

        seed_sim = self.attr['seed_sim']

        version = self.attr['version']

        maxfun = self.attr['maxfun']

        tau = self.attr['tau']

        # Number of parameters
        assert isinstance(num_paras, int)
        assert num_paras >= 53

        # Parallelism
        assert isinstance(num_procs, int)
        assert (num_procs > 0)
        if num_procs > 1:
            assert (version == 'FORTRAN')
            assert IS_PARALLEL

        # Version version of package
        assert (version in ['FORTRAN', 'PYTHON'])
        if version == 'FORTRAN':
            assert IS_FORTRAN

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

        # Number of draws for Monte Carlo integration
        assert (np.isfinite(num_draws_emax))
        assert (isinstance(num_draws_emax, int))
        assert (num_draws_emax >= 0)

        # Debugging mode
        assert (is_debug in [True, False])

        # Window for smoothing parameter
        assert (isinstance(tau, float))
        assert (tau > 0)

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
        assert (optimizer_used in OPT_EST_FORT + OPT_EST_PYTH)

        # Scaling
        assert (precond_spec['type'] in ['identity', 'gradient', 'magnitudes'])
        for key_ in ['minimum', 'eps']:
            assert (isinstance(precond_spec[key_], float))
            assert (precond_spec[key_] > 0.0)

        # Education
        assert isinstance(edu_spec['max'], int)
        assert edu_spec['max'] > 0
        assert isinstance(edu_spec['start'], list)
        assert len(edu_spec['start']) == len(set(edu_spec['start']))
        assert all(isinstance(item, int) for item in edu_spec['start'])
        assert all(item > 0 for item in edu_spec['start'])
        assert all(item <= edu_spec['max'] for item in edu_spec['start'])
        assert all(isinstance(item, float) for item in edu_spec['share'])
        assert all(0 <= item <= 1 for item in edu_spec['lagged'])
        assert all(0 <= item <= 1 for item in edu_spec['share'])
        np.testing.assert_almost_equal(
            np.sum(edu_spec['share']), 1.0, decimal=4)

        # Derivatives
        assert (derivatives in ['FORWARD-DIFFERENCES'])

        # Check model parameters
        check_model_parameters(optim_paras)

        # Check that all parameter values are within the bounds.
        x = get_optim_paras(optim_paras, num_paras, 'all', True)

        # It is not clear at this point how to impose parameter constraints on
        # the covariance matrix in a flexible manner. So, either all fixed or
        # none. As a special case, we also allow for all off-diagonal elements
        # to be fixed to zero.
        shocks_coeffs = optim_paras['shocks_cholesky'][np.tril_indices(4)]
        shocks_fixed = optim_paras['paras_fixed'][43:53]

        all_fixed = all(is_fixed is False for is_fixed in shocks_fixed)
        all_free = all(is_free is True for is_free in shocks_fixed)

        subset_fixed = [shocks_fixed[i] for i in [1, 3, 4, 6, 7, 8]]
        subset_value = [shocks_coeffs[i] for i in [1, 3, 4, 6, 7, 8]]

        off_diagonal_fixed = all(is_free is True for is_free in subset_fixed)
        off_diagonal_value = all(value == 0.0 for value in subset_value)
        off_diagonal = off_diagonal_fixed and off_diagonal_value

        if not (all_free or all_fixed or off_diagonal):
            raise UserError(' Misspecified constraints for covariance matrix')

        # Discount rate and type shares need to be larger than on at all times.
        for label in ['paras_fixed', 'paras_bounds']:
            assert isinstance(optim_paras[label], list)
            assert (len(optim_paras[label]) == num_paras)

        for i in range(1):
            assert optim_paras['paras_bounds'][i][0] >= 0.00

        for i in range(num_paras):
            lower, upper = optim_paras['paras_bounds'][i]
            if lower is not None:
                assert isinstance(lower, float)
                assert lower <= x[i]
                assert abs(lower) < PRINT_FLOAT
            if upper is not None:
                assert isinstance(upper, float)
                assert upper >= x[i]
                assert abs(upper) < PRINT_FLOAT
            if (upper is not None) and (lower is not None):
                assert upper >= lower
            # At this point no bounds for the elements of the covariance matrix
            # are allowed.
            if i in range(43, 53):
                assert optim_paras['paras_bounds'][i] == [None, None]

    def _check_integrity_results(self):
        """Check the integrity of the results."""
        # Check if solution attributes well maintained.
        for label in SOLUTION_ATTR:
            if self.attr['is_solved']:
                assert (self.attr[label] is not None)
            else:
                assert (self.attr[label] is None)

        # Distribute class attributes
        num_initial = len(self.attr['edu_spec']['start'])

        # We need to carefully distinguish between the maximum level of
        # schooling individuals enter the model and the maximum level they can
        # attain.
        edu_start = self.attr['edu_spec']['start']

        edu_start_max = max(edu_start)

        edu_max = self.attr['edu_spec']['max']

        num_periods = self.attr['num_periods']

        num_types = self.attr['num_types']

        # Distribute results
        periods_rewards_systematic = self.attr['periods_rewards_systematic']

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
        if periods_rewards_systematic is not None:
            periods_rewards_systematic = replace_missing_values(
                periods_rewards_systematic)
        if periods_emax is not None:
            periods_emax = replace_missing_values(periods_emax)

        # Check the creation of the state space
        is_applicable = (states_all is not None)
        is_applicable = is_applicable and (states_number_period is not None)
        is_applicable = is_applicable and (mapping_state_idx is not None)

        if is_applicable:
            # No values can be larger than constraint time. The exception in
            # the lagged schooling variable in the first period, which takes
            # value one but has index zero.
            for period in range(num_periods):
                assert (np.nanmax(states_all[period, :, :3]) <=
                        (period + edu_start_max))

            # Lagged schooling can only take value zero or one if finite.
            for period in range(num_periods):
                assert (np.nanmax(states_all[period, :, 3]) in [1, 2, 3, 4])
                assert (np.nanmin(states_all[period, :, :3]) == 0)

            # All finite values have to be larger or equal to zero.
            # The loop is required as np.all evaluates to FALSE for this
            # condition (see NUMPY documentation).
            for period in range(num_periods):
                assert (np.all(
                    states_all[period, :states_number_period[period]] >= 0))

            # The maximum of education years is never larger than `edu_max'.
            for period in range(num_periods):
                assert (np.nanmax(states_all[period, :, :][:, 2], axis=0) <=
                        edu_max)

            # Check for duplicate rows in each period. We only have possible
            # duplicates if there are multiple initial conditions.
            for period in range(num_periods):
                nstates = states_number_period[period]
                assert (np.sum(pd.DataFrame(
                        states_all[period, :nstates, :]).duplicated()) == 0)

            # Checking validity of state space values. All valid values need
            # to be finite.
            for period in range(num_periods):
                assert (np.all(np.isfinite(
                    states_all[period, :states_number_period[period]])))

            # There are no infinite values in final period.
            assert (np.all(np.isfinite(states_all[(num_periods - 1), :, :])))

            # Check the number of states in the first time period.
            num_states_start = num_types * num_initial * 2
            assert (np.sum(np.isfinite(
                mapping_state_idx[0, :, :, :, :])) == num_states_start)

            # Check that mapping is defined for all possible realizations of
            # the state space by period. Check that mapping is not defined for
            # all inadmissible values.
            is_infinite = np.tile(False, reps=mapping_state_idx.shape)
            for period in range(num_periods):
                # Subsetting valid indices
                nstates = states_number_period[period]
                indices = states_all[period, :nstates, :].astype('int')
                for index in indices:
                    # Check for finite value at admissible state
                    assert (np.isfinite(mapping_state_idx[period, index[0],
                            index[1], index[2], index[3] - 1, index[4]]))
                    # Record finite value
                    is_infinite[
                        period, index[0], index[1], index[2],
                        index[3] - 1, index[4]] = True
            # Check that all admissible states are finite
            assert (
                np.all(np.isfinite(mapping_state_idx[is_infinite == True])))

            # Check that all inadmissible states are infinite
            assert (np.all(np.isfinite(
                mapping_state_idx[is_infinite == False])) == False)

        # Check the calculated systematic rewards
        is_applicable = (states_all is not None)
        is_applicable = is_applicable and (states_number_period is not None)
        is_applicable = \
            is_applicable and (periods_rewards_systematic is not None)

        if is_applicable:
            # Check that the rewards are finite for all admissible values and
            # infinite for all others.
            is_infinite = np.tile(False, reps=periods_rewards_systematic.shape)
            for period in range(num_periods):
                # Loop over all possible states
                for k in range(states_number_period[period]):
                    # Check for finite value at admissible state
                    assert (np.all(np.isfinite(
                            periods_rewards_systematic[period, k, :])))
                    # Record finite value
                    is_infinite[period, k, :] = True
                # Check that all admissible states are finite
                assert (np.all(np.isfinite(
                        periods_rewards_systematic[is_infinite == True])))
                # Check that all inadmissible states are infinite
                if num_periods > 1:
                    assert (np.all(np.isfinite(
                        periods_rewards_systematic[
                            is_infinite == False])) == False)

        # Check the expected future value
        is_applicable = (periods_emax is not None)

        if is_applicable:
            # Check that the emaxs are finite for all admissible values and
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
                    assert (np.all(np.isfinite(
                            periods_emax[is_infinite == False])) == False)

    def _check_key(self, key):
        """Check that key is present."""
        assert (key in self.attr.keys()), \
            'Invalid key requested: {}'.format(key)
