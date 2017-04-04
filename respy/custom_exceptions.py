class MaxfunError(Exception):
    """ This custom-error class allows to enforce the MAXFUN restriction
    independent of the optimizer used.
    """
    pass

class UserError(Exception):
    """ This custom error class provides informative feedback in case of a misspecified request
    by the user.
    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return '\n\n         {}\n\n'.format(self.msg)
