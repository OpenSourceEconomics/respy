""" This module serves as the interface to the FORTRAN implementations of selected function.
"""

# Check for FORTRAN library
is_accelerated = True
try:
    import robupy.fort.fortran_functions as fort
except ImportError:
    is_accelerated = False

# project library
import robupy.fort.python_functions as py

# Replacements
if is_accelerated:
    calculate_payoffs_ex_ante = fort.calculate_payoffs_ex_ante
    get_future_payoffs = fort.get_future_payoffs
    simulate_emax = fort.simulate_emax
else:
    calculate_payoffs_ex_ante = py.calculate_payoffs_ex_ante
    get_future_payoffs = py.get_future_payoffs
    simulate_emax = py.simulate_emax