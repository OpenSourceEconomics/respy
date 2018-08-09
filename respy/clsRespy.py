import pickle as pkl
import numpy as np
import json
import copy

from respy.pre_processing.model_processing import write_init_file
from respy.python.shared.shared_auxiliary import dist_econ_paras
from respy.python.shared.shared_auxiliary import check_model_parameters
from respy.python.shared.shared_constants import ROOT_DIR
from respy.pre_processing.model_processing import read_init_file, convert_init_dict_to_attr_dict, \
    convert_attr_dict_to_init_dict
from respy.pre_processing.model_checking import check_model_attributes, \
    check_model_solution


class RespyCls(object):
    """Class that defines a model in respy.  """

    def __init__(self, fname):
        self._set_hardcoded_attributes()
        ini = read_init_file(fname)
        self.attr = convert_init_dict_to_attr_dict(ini)
        self._update_derived_attributes()
        self._initialize_solution_attributes()
        self.attr['is_locked'] = False
        self.attr['is_solved'] = False
        self.lock()

    def _set_hardcoded_attributes(self):
        """Set attributes that can't be changed by the model specification."""
        self.derived_attributes = ['is_myopic', 'num_paras']
        self.solution_attributes = [
            'periods_rewards_systematic', 'states_number_period',
            'mapping_state_idx', 'periods_emax', 'states_all']

    def _initialize_solution_attributes(self):
        """Initialize solution attributes to None."""
        for attribute in self.solution_attributes:
            self.attr[attribute] = None

    def update_optim_paras(self, x_econ):
        """Update model parameters."""
        x_econ = copy.deepcopy(x_econ)

        self.reset()

        delta, coeffs_common, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, \
            shocks_cov, type_shares, type_shifts = dist_econ_paras(x_econ)

        shocks_cholesky = np.linalg.cholesky(shocks_cov)
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
        check_model_parameters(optim_paras)

    def lock(self):
        """Lock class instance."""
        assert (not self.attr['is_locked']), \
            'Only unlocked instances of clsRespy can be locked.'

        self._update_derived_attributes()
        self._check_model_attributes()
        self._check_model_solution()
        self.attr['is_locked'] = True

    def unlock(self):
        """Unlock class instance."""
        assert self.attr['is_locked'], \
            'Only locked instances of clsRespy can be unlocked.'

        self.attr['is_locked'] = False

    def get_attr(self, key):
        """Get attributes."""
        assert self.attr['is_locked']
        self._check_key(key)

        if key in self.solution_attributes:
            assert self.get_attr('is_solved'), 'invalid request'

        return self.attr[key]

    def set_attr(self, key, value):
        """Set attributes."""
        assert (not self.attr['is_locked'])
        self._check_key(key)

        invalid_attr = self.derived_attributes + ['optim_paras', 'init_dict']
        if key in invalid_attr:
            raise AssertionError(
                '{} must not be modified by users!'.format(key))

        if key in self.solution_attributes:
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
        init_dict = convert_attr_dict_to_init_dict(self.attr)
        write_init_file(init_dict, fname)

    def reset(self):
        """Remove solution attributes from class instance."""
        for label in self.solution_attributes:
            self.attr[label] = None
        self.attr['is_solved'] = False

    def check_equal_solution(self, other):
        """Compare two class instances for equality of solution attributes."""
        assert (isinstance(other, RespyCls))

        for key_ in self.solution_attributes:
            try:
                np.testing.assert_almost_equal(
                    self.attr[key_], other.attr[key_])
            except AssertionError:
                return False

        return True

    def _update_derived_attributes(self):
        """Update derived attributes."""
        num_types = self.attr['num_types']
        self.attr['is_myopic'] = (self.attr['optim_paras']['delta'] == 0.00)[0]
        self.attr['num_paras'] = 53 + (num_types - 1) * 6

    def _check_model_attributes(self):
        """Check integrity of class instance.

        This testing is done the first time the class is locked and if
        the package is running in debug mode.

        """
        with open(ROOT_DIR + '/.config', 'r') as infile:
            config_dict = json.load(infile)

        check_model_attributes(self.attr, config_dict)

    def _check_model_solution(self):
        """Check the integrity of the results."""
        check_model_solution(self.attr)

    def _check_key(self, key):
        """Check that key is present."""
        assert (key in self.attr.keys()), \
            'Invalid key requested: {}'.format(key)
