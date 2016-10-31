import pandas as pd
import numpy as np

from respy.python.shared.shared_auxiliary import dist_model_paras


def write_info(respy_obj, data_frame):
    """ Write information about the simulated economy.
    """
    # Distribute class attributes
    file_sim = respy_obj.get_attr('file_sim')

    seed_sim = respy_obj.get_attr('seed_sim')

    # Get basic information
    num_agents_sim = data_frame[1].value_counts().iloc[0]

    num_periods = data_frame[0].value_counts().iloc[0]

    # Write information to file
    with open(file_sim + '.respy.info', 'w') as file_:

        file_.write('\n Simulated Economy\n\n')

        file_.write('   Number of Agents:       ' + str(num_agents_sim) + '\n\n')
        file_.write('   Number of Periods:      ' + str(num_periods) + '\n\n')
        file_.write('   Seed:                   ' + str(seed_sim) + '\n\n\n')
        file_.write('   Choices\n\n')

        fmt_ = '{:>10}' + '{:>14}' * 4 + '\n\n'
        labels = ['Period', 'Work A', 'Work B', 'Schooling', 'Home']
        file_.write(fmt_.format(*labels))

        for t in range(num_periods):

            work_a = np.sum((data_frame[2] == 1) &
                            (data_frame[1] == t))/float(num_agents_sim)

            work_b = np.sum((data_frame[2] == 2) & (data_frame[1] ==
                                                    t))/float(num_agents_sim)

            schooling = np.sum((data_frame[2] == 3) &
                               (data_frame[1] == t))/float(num_agents_sim)

            home = np.sum((data_frame[2] == 4) & (data_frame[1] ==
                                                  t))/float(num_agents_sim)

            fmt_ = '{:>10}' + '{:14.4f}' * 4 + '\n'
            args = [(t + 1), work_a, work_b, schooling, home]
            file_.write(fmt_.format(*args))

        file_.write('\n\n')
        file_.write('   Outcomes\n\n')

        for j, label in enumerate(['A', 'B']):

            file_.write('    Occupation ' + label + '\n\n')
            fmt_ = '{:>10}' + '{:>14}' * 6 + '\n\n'
            labels = [' Period', 'Counts',  'Mean', 'S.-Dev.',  '2. Decile']
            labels += ['5. Decile',  '8. Decile']
            file_.write(fmt_.format(*labels))

            for t in range(num_periods):

                is_working = (data_frame[2] == (j + 1)) & (data_frame[1] == t)
                wages = data_frame[is_working].ix[:, 3]
                count = wages.count()

                if count > 0:
                    mean, sd = np.mean(wages), np.sqrt(np.var(wages))
                    percentiles = np.percentile(wages, [20, 50, 80]).tolist()
                else:
                    mean, sd = '---', '---'
                    percentiles = ['---', '---', '---']

                values = [t + 1]
                values += [count, mean, sd]
                values += percentiles

                fmt_ = '{:>10}    ' + '{:>10}    ' * 6 + '\n'
                if count > 0:
                    fmt_ = '{:>10}    {:>10}' + '{:14.4f}' * 5 + '\n'
                file_.write(fmt_.format(*values))

            file_.write('\n')
        file_.write('\n')

        # Additional information about the simulated economy
        string = '''       {0[0]:<25}    {0[1]:10.4f}\n'''

        file_.write('   Additional Information\n\n')

        stat = data_frame[data_frame.ix[:, 1] ==
                (num_periods - 1)].ix[:, 6].mean()
        file_.write(string.format(['Average Education', stat]))

        file_.write('\n')

        stat = data_frame[data_frame.ix[:, 1] ==
                (num_periods - 1)].ix[:, 4].mean()
        file_.write(string.format(['Average Experience A', stat]))

        stat = data_frame[data_frame.ix[:, 1] ==
                (num_periods - 1)].ix[:, 5].mean()
        file_.write(string.format(['Average Experience B', stat]))

        file_.write('\n\n   Economic Parameters\n\n')
        fmt_ = '\n   {0:>10}' + '    {1:>25}\n\n'
        file_.write(fmt_.format(*['Identifier', 'Value']))
        # Write out the parametrization of the simulated economy.
        model_paras = respy_obj.get_attr('model_paras')
        vector = get_estimation_vector(model_paras, True)
        fmt_ = '   {:>10}' + '    {:25.5f}\n'
        for i, stat in enumerate(vector):
            file_.write(fmt_.format(*[i, stat]))


def write_out(respy_obj, data_frame):
    """ Write dataset to file.
    """
    # Distribute class attributes
    file_sim = respy_obj.get_attr('file_sim')

    formats = []

    formats += [_format_integer, _format_integer, _format_integer]

    formats += [_format_float, _format_integer, _format_integer]

    formats += [_format_integer, _format_integer]

    with open(file_sim + '.respy.dat', 'w') as file_:

        data_frame.to_string(file_, index=False, header=None, na_rep='.',
                            formatters=formats)


def _format_float(x):
    """ Pretty formatting for floats
    """
    if pd.isnull(x):
        return '    .'
    else:
        return '{0:10.2f}'.format(x)


def _format_integer(x):
    """ Pretty formatting for integers.
    """
    if pd.isnull(x):
        return '    .'
    else:
        return '{0:<5}'.format(int(x))


def get_estimation_vector(model_paras, is_debug):
    """ Construct the vector estimation arguments.
    """

    # Auxiliary objects
    shocks_cholesky = dist_model_paras(model_paras, is_debug)[-1]

    # Collect parameters
    vector = list()

    vector += model_paras['level'].tolist()

    vector += model_paras['coeffs_a'].tolist()

    vector += model_paras['coeffs_b'].tolist()

    vector += model_paras['coeffs_edu'].tolist()

    vector += model_paras['coeffs_home'].tolist()

    vector += shocks_cholesky[0, :1].tolist()

    vector += shocks_cholesky[1, :2].tolist()

    vector += shocks_cholesky[2, :3].tolist()

    vector += shocks_cholesky[3, :4].tolist()

    # Type conversion
    vector = np.array(vector)

    # Finishing
    return vector


def check_dataset_sim(data_frame, respy_obj):
    """ This routine runs some consistency checks on the simulated data frame.
    """
    # Distribute class attributes
    num_periods = respy_obj.get_attr('num_periods')

    edu_max = respy_obj.get_attr('edu_max')

    num_agents = respy_obj.get_attr('num_agents_sim')

    # Check dimension of data frame
    assert (data_frame.shape == (num_periods * num_agents, 8))

    # Check that there are the appropriate number of values
    for i in range(8):
        # This is not applicable for the wage observations
        if i == 3:
            continue
        # Check valid values
        assert (data_frame.count(axis=0)[i] == (num_periods * num_agents))

    # Check that the maximum number of values are valid
    for i, max_ in enumerate(
            [num_agents, num_periods, 4, num_periods, num_periods,
                num_periods,
                edu_max, 1]):
        # Agent index
        if i == 1:
            assert (data_frame.max(axis=0)[i] == (max_ - 1))

        if i == 0:
            assert (data_frame.max(axis=0)[i] == (max_ - 1))

        # Choice observation
        if i == 2:
            assert (data_frame.max(axis=0)[i] <= max_)
        # Wage observations
        if i == 3:
            pass
        # Work experience A, B
        if i in [4, 5]:
            assert (data_frame.max(axis=0)[i] <= max_)
        # Education
        if i in [6]:
            assert (data_frame.max(axis=0)[i] <= max_)
        # Lagged Education
        if i in [7]:
            assert (data_frame.max(axis=0)[i] <= max_)

    # Each valid period indicator occurs as often as agents in the dataset.
    for period in range(num_periods):
        assert (data_frame[1].value_counts()[period] == num_agents)

    # Each valid agent indicator occurs as often as periods in the dataset.
    for agent in range(num_agents):
        assert (data_frame[0].value_counts().iloc[agent] == num_periods)

    # Check valid values of wage observations
    for count in range(num_agents * num_periods):
        is_working = (data_frame[2][count] in [1, 2])
        if is_working:
            assert (np.isfinite(data_frame[3][count]))
            assert (data_frame[3][count] > 0.00)
        else:
            assert (np.isfinite(data_frame[3][count]) == False)
