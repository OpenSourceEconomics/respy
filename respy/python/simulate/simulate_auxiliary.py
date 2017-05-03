import pandas as pd
import numpy as np
import os

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.process.process_auxiliary import check_dataset_est


def construct_transition_matrix(base_df):
    """ This method constructs the transition matrix.
    """
    df = base_df.copy(deep=True)
    df['Choice_Next'] = df.groupby(level='Identifier')['Choice'].shift(-1)
    args = []
    for label in ['Choice', 'Choice_Next']:
        args += [pd.Categorical(df[label], categories=range(1, 5))]
    tm = pd.crosstab(*args, normalize='index').as_matrix()

    return tm


def write_info(respy_obj, data_frame):
    """ Write information about the simulated economy.
    """
    # Distribute class attributes
    optim_paras, num_types, file_sim, seed_sim, edu_spec = dist_class_attributes(respy_obj,
        'optim_paras', 'num_types', 'file_sim', 'seed_sim', 'edu_spec')

    # Get basic information
    num_agents_sim = len(data_frame['Identifier'].unique())
    num_periods = len(data_frame['Period'].unique())
    num_obs = num_agents_sim * num_periods

    # Write information to file
    with open(file_sim + '.respy.info', 'w') as file_:

        file_.write('\n Simulated Economy\n\n')

        file_.write('   Number of Agents:       ' + str(num_agents_sim) + '\n\n')
        file_.write('   Number of Periods:      ' + str(num_periods) + '\n\n')
        file_.write('   Seed:                   ' + str(seed_sim) + '\n\n\n')
        file_.write('   Choices\n\n')

        fmt_ = '{:>10}' + '{:>14}' * 4 + '\n\n'
        labels = ['Period', 'Work A', 'Work B', 'School', 'Home']
        file_.write(fmt_.format(*labels))

        choices = data_frame['Choice']
        for t in range(num_periods):
            args = []
            for decision in [1, 2, 3, 4]:
                args += [(choices.loc[slice(None), t] == decision).sum()]
            args = [x / float(num_agents_sim) for x in args]

            fmt_ = '{:>10}' + '{:14.4f}' * 4 + '\n'
            file_.write(fmt_.format((t + 1), *args))

        # We also print out the transition matrix as it provides some
        # insights about the persistence of choices. However, we can only
        # compute this transition matrix if the number of periods is larger
        # than one.
        if num_periods > 1:
            file_.write('\n\n')
            file_.write('    Transition Matrix\n\n')
            fmt_ = '{:>10}' + '{:>14}' * 4 + '\n\n'
            labels = ['Work A', 'Work B', 'School', 'Home']
            file_.write(fmt_.format(*[''] + labels))

            tb = construct_transition_matrix(data_frame)
            for i in range(4):
                fmt_ = '    {:6}' + '{:14.4f}' * 4 + '\n'
                line = [labels[i]] + tb[i, :].tolist()
                file_.write(fmt_.format(*line))

            file_.write('\n\n')

        # Now we can turn to the outcome information.
        file_.write('   Outcomes\n\n')

        for j, label in enumerate(['A', 'B']):

            file_.write('    Occupation ' + label + '\n\n')
            fmt_ = '{:>10}' + '{:>14}' * 6 + '\n\n'

            labels = []
            labels += [' Period', 'Counts',  'Mean', 'S.-Dev.',  '2. Decile']
            labels += ['5. Decile',  '8. Decile']

            file_.write(fmt_.format(*labels))

            for t in range(num_periods):

                is_working = (choices.loc[slice(None), t] == j + 1)
                wages = data_frame['Wage'].loc[slice(None), t][is_working]
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

        dat = data_frame['Years_Schooling'].loc[slice(None), num_periods - 1]
        file_.write(string.format(['Average Education', dat.mean()]))
        file_.write('\n')

        dat = data_frame['Experience_A'].loc[slice(None), num_periods - 1]
        file_.write(string.format(['Average Experience A',  dat.mean()]))

        dat = data_frame['Experience_B'].loc[slice(None), num_periods - 1]
        file_.write(string.format(['Average Experience B', dat.mean()]))

        file_.write('\n\n   Type Shares\n\n')
        dat = data_frame['Type'].value_counts().to_dict()
        fmt_ = '   {:>10}' + '    {:25.5f}\n'
        for type_ in range(num_types):
            try:
                share = dat[type_] / float(num_obs)
                file_.write(fmt_.format(*[type_, share]))
            # Some types might not occur in the simulated dataset. Then these are not part of the
            # dictionary with the value counts.
            except KeyError:
                file_.write(fmt_.format(*[type_, 0.0]))

        file_.write('\n\n   Initial Schooling Shares\n\n')
        dat = data_frame['Years_Schooling'][:, 0].value_counts().to_dict()
        for start in edu_spec['start']:
            try:
                share = dat[start] / float(num_obs)
                file_.write(fmt_.format(*[start, share]))
            # Some types might not occur in the simulated dataset. Then these are not part of the
            # dictionary with the value counts.
            except KeyError:
                file_.write(fmt_.format(*[start, 0.0]))

        file_.write('\n\n   Economic Parameters\n\n')
        fmt_ = '\n   {0:>10}' + '    {1:>25}\n\n'
        file_.write(fmt_.format(*['Identifier', 'Value']))
        vector = get_estimation_vector(optim_paras)
        fmt_ = '   {:>10}' + '    {:25.5f}\n'
        for i, stat in enumerate(vector):
            file_.write(fmt_.format(*[i, stat]))


def write_out(respy_obj, data_frame):
    """ Write dataset to file.
    """
    # Distribute class attributes
    file_sim = respy_obj.get_attr('file_sim')

    # The wage variable is formatted for two digits precision only.
    formatter = dict()
    formatter['Wage'] = format_float

    with open(file_sim + '.respy.dat', 'w') as file_:
        data_frame.to_string(file_, index=False, header=True, na_rep='.',
            formatters=formatter)


def format_float(x):
    """ Pretty formatting for floats
    """
    if pd.isnull(x):
        return '    .'
    else:
        return '{0:10.2f}'.format(x)


def format_integer(x):
    """ Pretty formatting for integers.
    """
    if pd.isnull(x):
        return '    .'
    else:
        return '{0:<5}'.format(int(x))


def get_estimation_vector(optim_paras):
    """ Construct the vector estimation arguments.
    """
    # Auxiliary objects
    num_types = len(optim_paras['type_shares'])

    # Collect parameters
    vector = list()
    vector += optim_paras['delta'].tolist()
    vector += optim_paras['level'].tolist()
    vector += optim_paras['coeffs_a'].tolist()
    vector += optim_paras['coeffs_b'].tolist()
    vector += optim_paras['coeffs_edu'].tolist()
    vector += optim_paras['coeffs_home'].tolist()
    vector += optim_paras['shocks_cholesky'][0, :1].tolist()
    vector += optim_paras['shocks_cholesky'][1, :2].tolist()
    vector += optim_paras['shocks_cholesky'][2, :3].tolist()
    vector += optim_paras['shocks_cholesky'][3, :4].tolist()
    vector += optim_paras['type_shares'].tolist()

    for i in range(1, num_types):
        vector += optim_paras['type_shifts'][i, :].tolist()

    # Type conversion
    vector = np.array(vector)

    # Finishing
    return vector


def check_dataset_sim(data_frame, respy_obj):
    """ This routine runs some consistency checks on the simulated dataset.
    Some more restrictions are imposed on the simulated dataset than the
    observed data.
    """
    # Distribute class attributes
    num_agents = respy_obj.get_attr('num_agents_sim')
    num_periods = respy_obj.get_attr('num_periods')
    num_types = respy_obj.get_attr('num_types')

    # Some auxiliary functions for later
    def check_check_time_constant(group):
        np.testing.assert_equal(group['Type'].nunique(), 1)

    def check_number_periods(group):
        np.testing.assert_equal(group['Period'].count(), num_periods)

    # So, we run all checks on the observed dataset.
    check_dataset_est(data_frame, respy_obj)

    # Checks for PERIODS
    dat = data_frame['Period']
    np.testing.assert_equal(dat.max(), num_periods - 1)

    # Checks for IDENTIFIER
    dat = data_frame['Identifier']
    np.testing.assert_equal(dat.max(), num_agents - 1)

    # Checks for TYPES
    dat = data_frame['Type']
    np.testing.assert_equal(dat.max() <= num_types - 1, True)
    np.testing.assert_equal(dat.isnull().any(), False)
    data_frame.groupby('Identifier').apply(check_check_time_constant)

    # Check that there are not missing wage observations if an agent is working. Also, we check
    # that if an agent is not working, there also is no wage observation.
    is_working = data_frame['Choice'].isin([1, 2])

    dat = data_frame['Wage'][is_working]
    np.testing.assert_equal(dat.isnull().any(), False)

    dat = data_frame['Wage'][~ is_working]
    np.testing.assert_equal(dat.isnull().all(), True)

    # Check that there are no missing observations and we follow an agent each period.
    data_frame.groupby('Identifier').apply(check_number_periods)


def get_random_types(num_types, optim_paras, num_agents_sim, is_debug):
    """ This function provides random draws for the types, or reads them
    in from a file.
    """
    if is_debug and os.path.exists('.types.respy.test'):
        types = np.genfromtxt('.types.respy.test')
    else:
        probs = optim_paras['type_shares'] / np.sum(optim_paras['type_shares'])
        types = np.random.choice(range(num_types), p=probs, size=num_agents_sim)

    # If we only have one individual, we need to ensure that types are a vector.
    types = np.array(types, ndmin=1)

    return types


def get_random_edu_start(edu_spec, num_agents_sim, is_debug):
    """ This function provides random draws for the initial schooling level, or reads them in 
    from a file.
    """
    if is_debug and os.path.exists('.initial.respy.test'):
        edu_start = np.genfromtxt('.initial.respy.test')
    else:
        edu_start = np.random.choice(edu_spec['start'], p=edu_spec['share'], size=num_agents_sim)

    # If we only have one individual, we need to ensure that types are a vector.
    edu_start = np.array(edu_start, ndmin=1)

    return edu_start