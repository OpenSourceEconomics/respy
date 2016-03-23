# standard library
import pandas as pd

# project library
from robupy.simulate.simulate_auxiliary import simulate_sample

from robupy.shared.auxiliary import replace_missing_values

''' Main function
'''
def simulate_python(periods_payoffs_systematic, mapping_state_idx,
        periods_emax, num_periods, states_all, num_agents, edu_start,
        edu_max, delta, periods_draws_sims, is_python):
    """ Wrapper for PYTHON and F2PY implementation of sample simulation.
    """
    # Interface to core functions
    if is_python:
        data_frame = simulate_sample(num_agents, states_all,
            num_periods, mapping_state_idx, periods_payoffs_systematic,
            periods_draws_sims, edu_max, edu_start, periods_emax, delta)
    else:
        import robupy.fortran.f2py_library as f2py_library
        data_frame = f2py_library.wrapper_simulate_sample(num_agents,
            states_all, num_periods, mapping_state_idx,
            periods_payoffs_systematic, periods_draws_sims, edu_max, edu_start,
            periods_emax, delta)

    # Replace missing values
    data_frame = replace_missing_values(data_frame)

    # Create pandas data frame
    data_frame = pd.DataFrame(data_frame)

    # Finishing
    return data_frame