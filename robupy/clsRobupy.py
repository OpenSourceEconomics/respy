""" Module that contains the class that carries around information for the
    ROBUPY package
"""

# standard library
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

        self.attr['num_periods'] = None

        self.attr['num_agents'] = None

        self.attr['num_draws'] = None

        self.attr['edu_start'] = None

        self.attr['edu_max'] = None

        self.attr['shocks'] = None

        self.attr['delta'] = None

        self.attr['debug'] = None

        self.attr['fast'] = None

        # Ambiguity
        self.attr['measure'] = None

        self.attr['level'] = None

        # Results
        self.attr['periods_payoffs_ex_post'] = None

        self.attr['states_number_period'] = None

        self.attr['mapping_state_idx'] = None

        self.attr['periods_emax'] = None

        self.attr['eps_cholesky'] = False

        self.attr['states_all'] = None

        self.attr['is_solved'] = False

        # The ex post realizations are only stored for debugging purposes.
        # In the special case of no randomness, they have to be equal to the
        # ex ante version. The same is true for the future payoffs
        self.attr['periods_payoffs_ex_ante'] = None

        self.attr['periods_future_payoffs'] = None

        # Status indicator
        self.is_locked = False

        self.is_first = True

        # This indicator is only used to compare the ROBUPY package to the
        # RESTUD codes. If set to true, it uses disturbances written out by
        # the RESTUD program. It aligns the random components across the two
        # components. It is only used in the development process.
        self.is_restud = False

    ''' Derived attributes
    '''
    def _derived_attributes(self):
        """ Calculate derived attributes.
        """
        # Distribute class attributes
        init_dict = self.attr['init_dict']

        is_first = self.is_first

        # Extract information from initialization dictionary
        if is_first:

            self.attr['seed_simulation'] = init_dict['SIMULATION']['seed']

            self.attr['num_agents'] = init_dict['SIMULATION']['agents']

            self.attr['seed_solution'] = init_dict['SOLUTION']['seed']

            self.attr['num_periods'] = init_dict['BASICS']['periods']

            self.attr['measure'] = init_dict['AMBIGUITY']['measure']

            self.attr['edu_start'] = init_dict['EDUCATION']['start']

            self.attr['num_draws'] = init_dict['SOLUTION']['draws']

            self.attr['level'] = init_dict['AMBIGUITY']['level']

            self.attr['edu_max'] = init_dict['EDUCATION']['max']

            self.attr['debug'] = init_dict['SOLUTION']['debug']

            self.attr['delta'] = init_dict['BASICS']['delta']

            self.attr['fast'] = init_dict['SOLUTION']['fast']

            self.attr['shocks'] = init_dict['SHOCKS']

            # Update status indicator
            self.is_first = False

    def _check_integrity(self):
        """ Check integrity of class instance. This testing is done the first
        time the class is locked and if the package is running in debug mode.
        """
        # Check applicability
        if (not self.is_first) and (not self.attr['debug']):
            return

        # Distribute class attributes
        seed_simulation = self.attr['seed_simulation']

        seed_solution = self.attr['seed_solution']

        num_agents = self.attr['num_agents']

        num_periods = self.attr['num_periods']

        edu_start = self.attr['edu_start']

        num_draws = self.attr['num_draws']

        measure = self.attr['measure']

        edu_max = self.attr['edu_max']

        shocks = self.attr['shocks']

        debug = self.attr['debug']

        delta = self.attr['delta']

        level = self.attr['level']

        fast = self.attr['fast']

        is_first = self.is_first

        # Debug status
        assert (debug in [True, False])

        # Constraints
        with_ambiguity = (level > 0.00)
        if with_ambiguity:
            assert (fast is False)

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
        assert (debug in [True, False])

        # Discount factor
        assert (np.isfinite(delta))
        assert (isinstance(delta, float))
        assert (delta >= 0.00)

        # Fast version of package
        assert (fast in [True, False])
        if fast:
            import robupy.performance.fortran.fortran_core as fortran_core

        # Shock distribution
        assert (isinstance(shocks, list))
        assert (np.all(np.isfinite(shocks)))
        assert (np.array(shocks).shape == (4, 4))

