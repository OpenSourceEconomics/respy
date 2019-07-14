import json
import os
import random
import socket
import string
import sys
import warnings
from pathlib import Path

import notifiers

RECIPIENTS = {"socrates": "janos.gabler@gmail.com", "abacus": "tobiasraabe@uni-bonn.de"}


def send_notification(which, **kwargs):
    """ Finishing up a run of the testing battery.
    """
    is_failed = None

    # Distribute keyword arguments
    if "is_failed" in kwargs.keys():
        is_failed = kwargs["is_failed"]

    if "idx_failures" in kwargs.keys():
        idx_failures = ", ".join(str(e) for e in kwargs["idx_failures"])

    hostname = socket.gethostname()

    if which == "regression":
        subject = " RESPY: Regression Testing"
        if is_failed:
            message = (
                "Failure during regression testing @"
                + hostname
                + " for test(s): "
                + idx_failures
                + "."
            )
        else:
            message = " Regression testing is completed on @" + hostname + "."

    else:
        raise AssertionError

    recipient = RECIPIENTS.get(hostname, "eisenhauer@policy-lab.org")

    # This allows to run the scripts even when no notification can be send.
    home = Path(os.environ.get("HOME") or os.environ.get("HOMEPATH"))
    credentials = home / ".credentials"

    if not credentials.exists():
        warnings.warn("No configuration file for notifications available.")
        sys.exit(0)

    credentials = json.loads(credentials.read_text())

    gmail = notifiers.get_notifier("gmail")
    gmail.notify(
        subject=subject,
        message=message,
        to=recipient,
        username=credentials["username"],
        password=credentials["password"],
    )


def get_random_dirname(length):
    """ This function creates a random directory name.

    The random name is used for a temporary testing directory. It starts with two
    underscores so that it does not clutter the root directory.

    TODO: Sensible length default.

    """
    return "__" + "".join(random.choice(string.ascii_lowercase) for _ in range(length))
