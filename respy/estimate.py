# project library
from respy.python.shared.shared_auxiliary import check_dataset
from respy.python.monitoring.clsMonitor import MonitorCls
from respy.python.process.process_python import process
from respy.fortran.interface import resfort_interface
from respy.python.interface import respy_interface


def estimate(respy_obj):
    """ Estimate the model
    """

    # Initialize monitoring subprocess, which provides the information about
    # the progress of the estimation independent of program version and
    # optimizer.
    monitor_obj = MonitorCls()
    monitor_obj.start()

    # Read in estimation dataset. It only reads in the number of agents
    # requested for the estimation.
    data_frame = process(respy_obj)
    data_array = data_frame.as_matrix()

    # Antibugging.
    assert _check_input(respy_obj, data_frame)

    # Distribute class attributes
    version = respy_obj.get_attr('version')

    # Select appropriate interface
    if version in ['PYTHON']:
        respy_interface(respy_obj, 'estimate', data_array)
    elif version in ['FORTRAN']:
        resfort_interface(respy_obj, 'estimate', data_array)
    else:
        raise NotImplementedError

    x, val = monitor_obj.stop()

    # Finishing
    return x, val


def _check_input(respy_obj, data_frame):
    """ Check input arguments.
    """
    # Check that class instance is locked.
    assert respy_obj.get_attr('is_locked')

    if respy_obj.get_attr('is_solved'):
        respy_obj.reset()

    # Check that dataset aligns with model specification.
    check_dataset(data_frame, respy_obj, 'est')

    # Special treatment required for the FORT optimizers. Since they are
    # written to the FORTRAN initialization file, we need a valid request for
    # all optimizers internally.
    optimizer_options = respy_obj.get_attr('optimizer_options')
    optimizer_used = respy_obj.get_attr('optimizer_used')

    assert (optimizer_used in optimizer_options.keys())

    # Now we check the integrity for all the optimizers that we find.
    if 'FORT-NEWUOA' in optimizer_options.keys():

        maxfun = optimizer_options['FORT-NEWUOA']['maxfun']
        rhobeg = optimizer_options['FORT-NEWUOA']['rhobeg']
        rhoend = optimizer_options['FORT-NEWUOA']['rhoend']
        npt = optimizer_options['FORT-NEWUOA']['npt']

        for var in [maxfun, npt]:
            assert isinstance(var, int)
            assert (var > 0)

        for var in [rhobeg, rhoend]:
            assert isinstance(var, float)
            assert (var > 0)

        assert (rhobeg > rhoend)

    if 'FORT-BFGS' in optimizer_options.keys():

        maxiter = optimizer_options['FORT-BFGS']['maxiter']
        epsilon = optimizer_options['FORT-BFGS']['epsilon']
        stpmx = optimizer_options['FORT-BFGS']['stpmx']
        gtol = optimizer_options['FORT-BFGS']['gtol']

        assert isinstance(maxiter, int)
        assert (maxiter > 0)

        for var in [stpmx, epsilon, gtol]:
            assert isinstance(var, float)
            assert (var > 0)

    if 'SCIPY-BFGS' in optimizer_options.keys():

        maxiter = optimizer_options['SCIPY-BFGS']['maxiter']
        epsilon = optimizer_options['SCIPY-BFGS']['epsilon']
        gtol = optimizer_options['SCIPY-BFGS']['gtol']

        assert isinstance(maxiter, int)
        assert (maxiter > 0)

        assert isinstance(gtol, float)
        assert (gtol > 0)

        assert isinstance(epsilon, float)
        assert (epsilon > 0)

    if 'SCIPY-POWELL' in optimizer_options.keys():

        maxfun = optimizer_options['SCIPY-POWELL']['maxfun']
        xtol = optimizer_options['SCIPY-POWELL']['xtol']
        ftol = optimizer_options['SCIPY-POWELL']['ftol']

        assert isinstance(maxfun, int)
        assert (maxfun > 0)

        assert isinstance(xtol, float)
        assert (xtol > 0)

        assert isinstance(ftol, float)
        assert (ftol > 0)

    # Finishing
    return respy_obj