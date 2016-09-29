import os

from respy.python.shared.shared_auxiliary import generate_optimizer_options
from respy.python.shared.shared_auxiliary import process_est_log
from respy.python.shared.shared_auxiliary import check_dataset
from respy.python.shared.shared_constants import OPT_EST_FORT
from respy.python.shared.shared_constants import OPT_AMB_FORT
from respy.python.shared.shared_constants import OPT_EST_PYTH
from respy.python.shared.shared_constants import OPT_AMB_PYTH
from respy.python.process.process_python import process
from respy.fortran.interface import resfort_interface
from respy.python.interface import respy_interface

OPTIMIZERS = OPT_EST_FORT + OPT_AMB_FORT + OPT_AMB_PYTH + OPT_EST_PYTH


def estimate(respy_obj):
    """ Estimate the model
    """
    # Cleanup
    for fname in ['est.respy.log', 'est.respy.info']:
        if os.path.exists(fname):
            os.unlink(fname)

    if respy_obj.get_attr('is_solved'):
        respy_obj.reset()

    assert check_estimation(respy_obj)

    # Read in estimation dataset. It only reads in the number of agents
    # requested for the estimation.
    data_frame = process(respy_obj)
    data_array = data_frame.as_matrix()

    # Distribute class attributes
    version = respy_obj.get_attr('version')

    # Select appropriate interface
    if version in ['PYTHON']:
        respy_interface(respy_obj, 'estimate', data_array)
    elif version in ['FORTRAN']:
        resfort_interface(respy_obj, 'estimate', data_array)
    else:
        raise NotImplementedError

    rslt = process_est_log()
    x, val = rslt['paras_final'], rslt['value_final']

    # Finishing
    return x, val


def check_estimation(respy_obj):
    """ Check input arguments.
    """
    # Check that class instance is locked.
    assert respy_obj.get_attr('is_locked')

    # Distribute class attributes
    optimizer_options = respy_obj.get_attr('optimizer_options')
    optimizer_used = respy_obj.get_attr('optimizer_used')
    model_paras = respy_obj.get_attr('model_paras')
    paras_fixed = respy_obj.get_attr('paras_fixed')
    version = respy_obj.get_attr('version')
    maxfun = respy_obj.get_attr('maxfun')

    # Get auxiliary objects
    level = model_paras['level'][0]

    # Check that the used optimizers were defined by the user.
    if level > 0:
        if version == 'FORTRAN':
            assert 'FORT-SLSQP' in optimizer_options.keys()
        if version == 'PYTHON':
            assert 'SCIPY-SLSQP' in optimizer_options.keys()

    if maxfun > 0:
        assert optimizer_used in optimizer_options.keys()

        # When the level of ambiguity is a free parameter, then we can only
        # allow for the constraint optimizers in the estimation.
        if not paras_fixed[0]:
            if version == 'PYTHON':
                assert optimizer_used in ['SCIPY-LBFGSB']
                assert 'SCIPY-SLSQP' in optimizer_options.keys()
            else:
                assert optimizer_used in ['FORT-BOBYQA']
                assert 'FORT-SLSQP' in optimizer_options.keys()

    # We need to make sure that all optimizers are fully defined for the
    # FORTRAN interface. At the same time, we do not want to require the user
    # to specify only the optimizers that are used. So, we sample a full set
    # and replace the optimizers that are used with the user specification.
    full_options = dict()
    for optimizer in OPTIMIZERS:
        full_options[optimizer] = \
            generate_optimizer_options(optimizer, paras_fixed)

    for optimizer in optimizer_options.keys():
        full_options[optimizer] = optimizer_options[optimizer]

    # Update the enlarged set of optimizer options.
    check_optimizer_options(full_options, paras_fixed)

    respy_obj.unlock()
    respy_obj.set_attr('optimizer_options', full_options)
    respy_obj.lock()

    # Check that dataset aligns with model specification.
    data_frame = process(respy_obj)
    check_dataset(data_frame, respy_obj, 'est')

    # Finishing
    return respy_obj


def check_optimizer_options(optimizer_options, paras_fixed):
    """ This function makes sure that the optimizer options are all valid.
    """
    # POWELL's algorithms
    for optimizer in ['FORT-NEWUOA', 'FORT-BOBYQA']:
        maxfun = optimizer_options[optimizer]['maxfun']
        rhobeg = optimizer_options[optimizer]['rhobeg']
        rhoend = optimizer_options[optimizer]['rhoend']
        npt = optimizer_options[optimizer]['npt']

        for var in [maxfun, npt]:
            assert isinstance(var, int)
            assert (var > 0)
        for var in [rhobeg, rhoend]:
            assert (rhobeg > rhoend)
            assert isinstance(var, float)
            assert (var > 0)

    # FORT-BFGS
    maxiter = optimizer_options['FORT-BFGS']['maxiter']
    stpmx = optimizer_options['FORT-BFGS']['stpmx']
    gtol = optimizer_options['FORT-BFGS']['gtol']
    assert isinstance(maxiter, int)
    assert (maxiter > 0)
    for var in [stpmx, gtol]:
        assert isinstance(var, float)
        assert (var > 0)

    # FORT-SLSQP
    maxiter = optimizer_options['FORT-SLSQP']['maxiter']
    ftol = optimizer_options['FORT-SLSQP']['ftol']
    eps = optimizer_options['FORT-SLSQP']['eps']
    assert isinstance(maxiter, int)
    assert (maxiter > 0)
    for var in [eps, ftol]:
        assert isinstance(var, float)
        assert (var > 0)

    # SCIPY-BFGS
    maxiter = optimizer_options['SCIPY-BFGS']['maxiter']
    gtol = optimizer_options['SCIPY-BFGS']['gtol']
    eps = optimizer_options['SCIPY-BFGS']['eps']
    assert isinstance(maxiter, int)
    assert (maxiter > 0)
    for var in [eps, gtol]:
        assert isinstance(var, float)
        assert (var > 0)

    # SCIPY-LBFGSB
    maxiter = optimizer_options['SCIPY-LBFGSB']['maxiter']
    pgtol = optimizer_options['SCIPY-LBFGSB']['pgtol']
    factr = optimizer_options['SCIPY-LBFGSB']['factr']
    maxls = optimizer_options['SCIPY-LBFGSB']['maxls']
    eps = optimizer_options['SCIPY-LBFGSB']['eps']
    m = optimizer_options['SCIPY-LBFGSB']['m']

    for var in [pgtol, factr, eps]:
        assert isinstance(var, float)
        assert var > 0
    for var in [m, maxiter, maxls]:
        assert isinstance(var, int)
        assert (var >= 0)

    # SCIPY-POWELL
    maxiter = optimizer_options['SCIPY-POWELL']['maxiter']
    maxfun = optimizer_options['SCIPY-POWELL']['maxfun']
    xtol = optimizer_options['SCIPY-POWELL']['xtol']
    ftol = optimizer_options['SCIPY-POWELL']['ftol']
    assert isinstance(maxiter, int)
    assert (maxiter > 0)
    assert isinstance(maxfun, int)
    assert (maxfun > 0)
    assert isinstance(xtol, float)
    assert (xtol > 0)
    assert isinstance(ftol, float)
    assert (ftol > 0)

    # SCIPY-SLSQP
    maxiter = optimizer_options['SCIPY-SLSQP']['maxiter']
    ftol = optimizer_options['SCIPY-SLSQP']['ftol']
    eps = optimizer_options['SCIPY-SLSQP']['eps']
    assert isinstance(maxiter, int)
    assert (maxiter > 0)
    for var in [eps, ftol]:
        assert isinstance(var, float)
        assert (var > 0)

