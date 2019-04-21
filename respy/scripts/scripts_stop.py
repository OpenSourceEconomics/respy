import argparse
import os

from respy.custom_exceptions import UserError


def stop():
    """ This function sends a signal to the package that the estimation is to be stopped
    immediately. It results in a gentle termination.
    """
    if os.path.exists(".estimation.respy.scratch"):
        open(".stop.respy.scratch", "w").close()
    else:
        raise UserError("... no estimation running at this time")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Stop estimation run of the RESPY package."
    )

    stop()
