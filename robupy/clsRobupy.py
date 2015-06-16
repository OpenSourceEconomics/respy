""" Module that contains the class that carries around information for the
    grmpy package
"""

# standard library
import pickle as pkl

# project library
from robupy.clsMeta import MetaCls

class RobupyCls(MetaCls):
    
    def __init__(self):
        
        self.attr = dict()

        self.attr['init_dict'] = None

        # Derived attributes
        self.attr['num_periods'] = None

        self.attr['num_agents'] = None

        self.attr['num_draws'] = None

        self.attr['edu_start'] = None

        self.attr['edu_max'] = None

        self.attr['delta'] = None

        self.attr['shocks'] = None

        self.attr['seed'] = None

        # Results
        self.attr['emax'] = None

        self.attr['k_period'] = None

        self.attr['k_state'] = None

        self.attr['payoffs_ex_ante'] = None

        self.attr['f_state'] = None

        # Status indicator
        self.is_locked = False

    ''' Derived attributes
    '''
    def _derived_attributes(self):
        """ Calculate derived attributes.
        """

        # Distribute class attributes
        init_dict = self.attr['init_dict']

        #
        self.attr['num_periods'] = init_dict['BASICS']['periods']

        self.attr['num_agents'] = init_dict['BASICS']['agents']

        self.attr['delta'] = init_dict['BASICS']['delta']


        self.attr['edu_start'] = init_dict['EDUCATION']['initial']

        self.attr['edu_max'] = init_dict['EDUCATION']['maximum']

        self.attr['num_draws'] = init_dict['COMPUTATION']['draws']

        self.attr['seed'] = init_dict['COMPUTATION']['seed']

        self.attr['shocks'] = init_dict['SHOCKS']

    # TODO: TESTING