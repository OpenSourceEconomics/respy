import json
import os
import socket
import sys
import warnings
from pathlib import Path

import notifiers

RECIPIENTS = {"socrates": "janos.gabler@gmail.com", "abacus": "tobiasraabe@uni-bonn.de"}


def send_notification(subject, message):
    """Send notification."""
    hostname = socket.gethostname()

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
