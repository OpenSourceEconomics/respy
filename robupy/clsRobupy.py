""" Module that contains the class that carries around information for the
    grmpy package
"""

# standard library
import pickle as pkl

# project library
from robupy.clsMeta import MetaCls

class RobupyCls(MetaCls):
    
    def __init__(self):
        """ Initialization of hand-crafted class for package management.
        """
        self.attr = dict()

        self.attr['init_dict'] = None

        # Derived attributes
        self.attr['seed_solution'] = None

        self.attr['seed_simulation'] = None

        self.attr['num_periods'] = None

        self.attr['num_agents'] = None

        self.attr['num_draws'] = None

        self.attr['edu_start'] = None

        self.attr['edu_max'] = None

        self.attr['delta'] = None

        self.attr['shocks'] = None

        self.attr['debug'] = None

        self.attr['fast'] = None

        # Ambiguity
        self.attr['measure'] = None

        self.attr['level'] = None

        # Results
        self.attr['periods_emax'] = None

        self.attr['states_number_period'] = None

        self.attr['states_all'] = None

        self.attr['periods_payoffs_ex_post'] = None

        self.attr['mapping_state_idx'] = None

        self.attr['is_solved'] = False

        # The ex post realizations are only stored for
        # debugging purposes. In the special case of
        # no randomness, they have to be equal to the
        # ex ante version. The same is true for the
        # future payoffs
        self.attr['periods_payoffs_ex_ante'] = None

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

        if is_first:

            self.attr['num_periods'] = init_dict['BASICS']['periods']

            self.attr['delta'] = init_dict['BASICS']['delta']

            self.attr['edu_start'] = init_dict['EDUCATION']['start']

            self.attr['edu_max'] = init_dict['EDUCATION']['max']

            self.attr['num_draws'] = init_dict['SOLUTION']['draws']

            self.attr['debug'] = init_dict['SOLUTION']['debug']

            self.attr['seed_solution'] = init_dict['SOLUTION']['seed']

            self.attr['seed_simulation'] = init_dict['SIMULATION']['seed']

            self.attr['fast'] = init_dict['SOLUTION']['fast']

            self.attr['num_agents'] = init_dict['SIMULATION']['agents']

            self.attr['shocks'] = init_dict['SHOCKS']

            self.attr['measure'] = init_dict['AMBIGUITY']['measure']

            self.attr['level'] = init_dict['AMBIGUITY']['level']

            # Update status indicator
            self.is_first = False

    def _check_integrity(self):
        """ Check integrity of class instance.
        """

        # Debug status
        assert (self.attr['debug'] in [True, False])

        # Constraints
        with_ambiguity = (self.attr['level'] > 0.00)
        if with_ambiguity:
            assert (self.attr['fast'] is False)

        # Check library
        if self.attr['fast']:
            import robupy.performance.fortran.fortran_core as fortran_core