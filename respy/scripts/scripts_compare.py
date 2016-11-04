#!/usr/bin/env python
from statsmodels.tools.eval_measures import rmse
import numpy as np
import os

from respy.python.simulate.simulate_auxiliary import format_float
from respy.python.process.process_python import process
from respy import RespyCls, simulate


def dist_input_arguments(parser):
    """ Check input for estimation script.
    """
    # Parse arguments
    args = parser.parse_args()

    # Distribute arguments
    request = args.request
    init = args.init

    # Check attributes
    assert (os.path.exists(init))

    # Finishing
    return request, init

def scripts_compare():
    init_file = "model.respy.ini"

    # Read in baseline model specification.
    respy_obj = RespyCls(init_file)

    # First we need to read in the empirical data

    data_obs = process(respy_obj)



    _, data_sim = simulate(respy_obj)


    def _prepare_outcomes(data_obs, data_sim, which):
        if which == 'Occupation A':
            choice_ind = 1
        else:
            choice_ind = 2
        num_periods = len(data_obs['Period'].unique())

        rslt = dict()
        rslt['Observed'] = []
        rslt['Simulated'] = []

        for label in ['Observed', 'Simulated']:
            if label == 'Observed':
                data = data_obs
            else:
                data = data_sim

            for period in range(num_periods):
                is_occupation = data['Choice'] == choice_ind
                series = data['Earnings'].ix[is_occupation][:, period]
                rslt[label] += [list(series.describe().values)]
        return rslt


    def _prepare_choices(data_obs, data_sim):
        """ This function prepares the information about the choice probabilities
        for easy printing.
        """
        num_periods = len(data_obs['Period'].unique())

        rslt_full = dict()
        rslt_full['Observed'] = []
        rslt_full['Simulated'] = []


        rslt_shares = dict()
        rslt_shares['Observed'] = []
        rslt_shares['Simulated'] = []

        for label in ['Observed', 'Simulated']:
            if label == 'Observed':
                data = data_obs
            else:
                data = data_sim
            for period in range(num_periods):
                shares = []
                total = data['Choice'].loc[:, period].count()
                for choice in [1, 2, 3, 4]:
                    count = np.sum(data['Choice'].loc[:, period] == choice)
                    shares += [count / float(total)]
                rslt_full[label] += [[total] + shares]
                rslt_shares[label] += shares

        # We also prepare the overall RMSE.
        rmse_choice = rmse(rslt_shares['Observed'], rslt_shares['Simulated'])

        return rslt_full, rmse_choice


    # Prepare results
    rslt_choice, rmse_choice = _prepare_choices(data_obs, data_sim)

    rslt_A = _prepare_outcomes(data_obs, data_sim, 'Occupation A')
    rslt_B = _prepare_outcomes(data_obs, data_sim, 'Occupation B')

    max_obs = len(data_obs['Period'].unique())

    with open('compare.respy.info', 'w') as file_:

        file_.write('\n Comparing the Observed and Simulated Economies\n\n')

        # Comparing the choice distributions
        file_.write('\n   Choices \n\n')
        fmt_ = '{:>15}' * 7 + '\n'
        labels = ['Data', 'Period', 'Count', 'White', 'Blue', 'School', 'Home']
        file_.write(fmt_.format(*labels) + '\n')
        for period in range(max_obs):
            for name in ['Observed', 'Simulated']:
                line = [name, period] + rslt_choice[name][period]
                fmt_ = '{:>15}' * 3 + '{:15.2f}' * 4 + '\n'
                file_.write(fmt_.format(*line))
            file_.write('\n')
        line = '   Overall RMSE {:14.5f}\n'.format(rmse_choice)
        file_.write(line)

        # Comparing the earnings distributions
        file_.write('\n\n   Outcomes \n\n')
        fmt_ = '{:>15}' * 8 + '\n'
        labels = ['Data', 'Period', 'Count', 'Mean', 'Std.', '25%', '50%', '75%']
        file_.write(fmt_.format(*labels) + '\n')
        for rslt, name in [(rslt_A, 'Occupation A'), (rslt_B, 'Occupation B')]:
            file_.write('\n    ' + name + ' \n\n')
            for period in range(max_obs):
                for label in ['Observed', 'Simulated']:
                    counts = int(rslt[label][period][0])
                    line = [label, period, counts]
                    # The occurance of NAN requires special care.
                    stats = rslt[label][period][1:]
                    stats = [format_float(x) for x in stats]
                    file_.write(fmt_.format(*line + stats))
                file_.write('\n')
