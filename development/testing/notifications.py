import json
import os
import socket
import sys
import warnings
from pathlib import Path

import apprise

RECIPIENTS = {"socrates": "janos.gabler@gmail.com", "abacus": "traabe92@gmail.com"}


def send_notification(title, body):
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
    message_header = {
        "domain": "gmail.com",
        "to": recipient,
        "name": "respy",
        **credentials,
    }
    service = "mailto://{username}:{password}@{domain}?to={to}&name={name}"

    apobj = apprise.Apprise()
    apobj.add(service.format(**message_header))
    apobj.notify(title=title, body=body)
