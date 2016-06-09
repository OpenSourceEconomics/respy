
def add_optimizers(respy_obj):

    optimizer_options = respy_obj.get_attr('optimizer_options')


    for optimizer in ['FORT-NEWUOA', 'FORT-BFGS']:

        # Skip if defined by user.
        if optimizer in optimizer_options.keys():
            continue

        if optimizer in ['FORT-NEWUOA']:
            optimizer_options[optimizer] = dict()
            optimizer_options[optimizer]['npt'] = 40
            optimizer_options[optimizer]['rhobeg'] = 0.1
            optimizer_options[optimizer]['rhoend'] = 0.0001
            optimizer_options[optimizer]['maxfun'] = 20

        if optimizer in ['FORT-BFGS']:
            optimizer_options[optimizer] = dict()
            optimizer_options[optimizer]['epsilon'] = 0.00001
            optimizer_options[optimizer]['gtol'] = 0.00001
            optimizer_options[optimizer]['maxiter'] = 10
            optimizer_options[optimizer]['stpmx'] = 100.0

    respy_obj.unlock()
    respy_obj.set_attr('optimizer_options', optimizer_options)
    respy_obj.lock()

    return respy_obj