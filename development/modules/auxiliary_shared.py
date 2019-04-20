import argparse
import os
import random
import socket
import string
import subprocess
import sys
from pathlib import Path
from string import Formatter

import numpy as np

from development.modules.clsMail import MailCls
from development.modules.config import PACKAGE_DIR


def update_class_instance(respy_obj, spec_dict):
    """ Update model specification from the baseline initialization file.
    """

    respy_obj.unlock()

    # Varying the baseline level of ambiguity requires special case. The same is true
    # for the discount rate.
    if "level" in spec_dict["update"].keys():
        respy_obj.attr["optim_paras"]["level"] = np.array(
            [spec_dict["update"]["level"]]
        )
    if "delta" in spec_dict["update"].keys():
        respy_obj.attr["optim_paras"]["delta"] = np.array(
            [spec_dict["update"]["delta"]]
        )

    for key_ in spec_dict["update"].keys():
        if key_ in ["level", "delta"]:
            continue
        respy_obj.set_attr(key_, spec_dict["update"][key_])

    respy_obj.lock()

    return respy_obj


def strfdelta(tdelta, fmt):
    """ Get a string from a timedelta.
    """
    f, d = Formatter(), {}
    l = {"D": 86400, "H": 3600, "M": 60, "S": 1}
    k = list(map(lambda x: x[1], list(f.parse(fmt))))
    rem = int(tdelta.total_seconds())
    for i in ("D", "H", "M", "S"):
        if i in k and i in l.keys():
            d[i], rem = divmod(rem, l[i])
    return f.format(fmt, **d)


def cleanup():
    os.system("git clean -d -f")


def compile_package(is_debug=False):
    """ Compile the package for use.
    """
    python_exec = sys.executable
    cwd = os.getcwd()
    os.chdir(PACKAGE_DIR + "/respy")
    subprocess.check_call(python_exec + " waf distclean", shell=True)
    if not is_debug:
        subprocess.check_call(python_exec + " waf configure build", shell=True)
    else:
        subprocess.check_call(python_exec + " waf configure build --debug ", shell=True)

    os.chdir(cwd)


def send_notification(which, **kwargs):
    """ Finishing up a run of the testing battery.
    """
    # This allows to run the scripts even when no notification can be send.
    home = Path(os.environ.get("HOME") or os.environ.get("HOMEPATH"))
    if not os.path.exists(str(home / ".credentials")):
        return

    hours, is_failed, num_tests, seed = None, None, None, None

    # Distribute keyword arguments
    if "is_failed" in kwargs.keys():
        is_failed = kwargs["is_failed"]

    if "hours" in kwargs.keys():
        hours = "{}".format(kwargs["hours"])

    if "procs" in kwargs.keys():
        procs = str(kwargs["procs"])

    if "num_tests" in kwargs.keys():
        num_tests = "{}".format(kwargs["num_tests"])

    if "seed" in kwargs.keys():
        seed = "{}".format(kwargs["seed"])

    if "idx_failures" in kwargs.keys():
        idx_failures = ", ".join(str(e) for e in kwargs["idx_failures"])

    if "old_release" in kwargs.keys():
        old_release = kwargs["old_release"]

    if "new_release" in kwargs.keys():
        new_release = kwargs["new_release"]

    if "failed_seeds" in kwargs.keys():
        failed_seeds = kwargs["failed_seeds"]

    hostname = socket.gethostname()

    if which == "scalability":
        subject = " RESPY: Scalability Testing"
        message = " Scalability testing is completed on @" + hostname + "."
    elif which == "reliability":
        subject = " RESPY: Reliability Testing"
        message = " Reliability testing is completed on @" + hostname + "."
    elif which == "property":
        subject = " RESPY: Property Testing"
        message = (
            " A "
            + hours
            + " hour run of the testing battery on @"
            + hostname
            + " is completed."
        )

    elif which == "regression":
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

    elif which == "release":
        subject = " RESPY: Release Testing"
        if is_failed:
            message = (
                " Failure during release testing with seed "
                + seed
                + " on @"
                + hostname
                + "."
            )
        else:
            message = (
                " Release testing completed successfully after "
                + hours
                + " hours on @"
                + hostname
                + ". We compared release "
                + old_release
                + " against "
                + new_release
                + " for a total of "
                + num_tests
                + " tests."
            )
    elif which == "robustness":
        subject = " RESPY: Robustness Testing"
        if is_failed is True:
            failed_pct = len(failed_seeds) / float(num_tests) * 100
            failed_pct = "({} %)".format(failed_pct)
            message = (
                " Failure during robustness testing on @{}. In total {} "
                + "tests were run in {} hours on {} cores. {} tests failed "
                + "{}. The failed tests have the following seeds: {}."
            )
            message = message.format(
                hostname,
                num_tests,
                hours,
                procs,
                len(failed_seeds),
                failed_pct,
                str(failed_seeds),
            )
        else:
            message = (
                " Robustness testing completed successfully  after "
                + hours
                + " hours, running on "
                + procs
                + " cores on @"
                + hostname
                + ". In total "
                + num_tests
                + " were run."
            )
    else:
        raise AssertionError

    mail_obj = MailCls()
    mail_obj.set_attr("subject", subject)
    mail_obj.set_attr("message", message)

    if which == "property":
        mail_obj.set_attr("attachment", "property.respy.info")
    elif which == "scalability":
        mail_obj.set_attr("attachment", "scalability.respy.info")
    elif which == "reliability":
        mail_obj.set_attr("attachment", "reliability.respy.info")
    elif which == "robustness":
        mail_obj.set_attr("attachment", "robustness.respy.info")

    mail_obj.lock()

    mail_obj.send()


def aggregate_information(which, fnames):
    """ This function simply aggregates the information from the different specifications.
    """
    if which == "scalability":
        fname_info = "scalability.respy.info"
    elif which == "reliability":
        fname_info = "reliability.respy.info"
    else:
        raise AssertionError

    dirnames = []
    for fname in fnames:
        dirnames += [fname.replace(".ini", "")]

    with open(fname_info, "w") as outfile:
        outfile.write("\n")
        for dirname in dirnames:

            if not os.path.exists(dirname):
                continue

            outfile.write(" " + dirname + "\n")
            os.chdir(dirname)
            with open(fname_info, "r") as infile:
                outfile.write(infile.read())
            os.chdir("../")
            outfile.write("\n\n")


def process_command_line_arguments(description):
    """ Process command line arguments for the request.
    """
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--debug",
        action="store_true",
        dest="is_debug",
        default=False,
        help="use debugging specification",
    )

    args = parser.parse_args()
    is_debug = args.is_debug

    return is_debug


def get_random_dirname(length):
    """ This function creates a random directory name.

    The random name is used for a temporary testing directory. It starts with two
    underscores so that it does not clutter the root directory.

    TODO: Sensible length default.

    """
    return "__" + "".join(random.choice(string.ascii_lowercase) for _ in range(length))
