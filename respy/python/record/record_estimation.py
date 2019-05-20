import time
from datetime import datetime

import numpy as np
import scipy

from respy.python.record.record_warning import record_warning
from respy.python.shared.shared_auxiliary import cholesky_to_coeffs
from respy.python.shared.shared_auxiliary import distribute_parameters
from respy.python.shared.shared_auxiliary import extract_cholesky
from respy.python.shared.shared_constants import LARGE_FLOAT


def record_estimation_scaling(
    x_optim_free_unscaled_start,
    x_optim_free_scaled_start,
    paras_bounds_free_scaled,
    precond_matrix,
    paras_fixed,
    is_setup,
):

    with open("est.respy.log", "a") as out_file:
        if is_setup:
            out_file.write(" {:}\n\n".format("PRECONDITIONING"))
            fmt_ = "   {:>10}" + "    {:>25}" * 5 + "\n\n"
            labels = ["Identifier", "Original", "Scale"]
            labels += ["Transformed Value", "Transformed Lower"]
            labels += ["Transformed Upper"]
            out_file.write(fmt_.format(*labels))
        else:
            j = 0
            for i, is_fixed in enumerate(paras_fixed):
                if is_fixed:
                    continue

                paras = [
                    i,
                    x_optim_free_unscaled_start[j],
                    precond_matrix[j, j],
                    x_optim_free_scaled_start[j],
                    paras_bounds_free_scaled[j, 0],
                    paras_bounds_free_scaled[j, 1],
                ]

                for k in [4, 5]:
                    if abs(paras[k]) > LARGE_FLOAT:
                        paras[k] = "---"
                    else:
                        paras[k] = "{:25.15f}".format(paras[k])

                fmt = "   {:>10}" + "    {:25.15f}" * 3 + "    {:>25}" * 2 + "\n"
                out_file.write(fmt.format(*paras))

                j += 1

            out_file.write("\n")


def record_estimation_stop():
    with open("est.respy.info", "a") as out_file:
        out_file.write("\n TERMINATED\n")


def record_estimation_eval(opt_obj, fval, x_optim_all_unscaled, start):
    """Log the progress of an estimation.

    This function contains two parts as two files provide information about the
    progress.

    """
    # Distribute class attributes
    paras_fixed = opt_obj.paras_fixed
    num_paras = opt_obj.num_paras
    num_types = opt_obj.num_types

    shocks_cholesky, _ = extract_cholesky(x_optim_all_unscaled, 0)
    shocks_coeffs = cholesky_to_coeffs(shocks_cholesky)

    # Identify events
    is_start = opt_obj.num_eval == 0
    is_step = opt_obj.crit_vals[1] > fval
    x_optim_shares = x_optim_all_unscaled[53 : 53 + (num_types - 1) * 2]

    for i in range(3):
        if i == 0 and not is_start:
            continue

        if i == 1:
            if not is_step:
                continue
            else:
                opt_obj.num_step += 1

        if i == 2:
            opt_obj.num_eval += 1

        opt_obj.crit_vals[i] = fval
        opt_obj.x_optim_container[:, i] = x_optim_all_unscaled
        opt_obj.x_econ_container[:43, i] = x_optim_all_unscaled[:43]
        opt_obj.x_econ_container[43:53, i] = shocks_coeffs
        opt_obj.x_econ_container[53 : 53 + (num_types - 1) * 2, i] = x_optim_shares
        opt_obj.x_econ_container[
            53 + (num_types - 1) * 2 : num_paras, i
        ] = x_optim_all_unscaled[53 + (num_types - 1) * 2 :]

    x_optim_container = opt_obj.x_optim_container
    x_econ_container = opt_obj.x_econ_container

    # Now we turn to est.respy.info
    with open("est.respy.log", "a") as out_file:
        fmt_ = " {0:>4}{1:>13}" + " " * 10 + "{2:>4}{3:>10}\n\n"
        line = ["EVAL", opt_obj.num_eval, "STEP", opt_obj.num_step]
        out_file.write(fmt_.format(*line))
        fmt_ = "   {0:<9}     {1:>25}\n"
        out_file.write(fmt_.format(*["Date", time.strftime("%d/%m/%Y")]))
        fmt_ = "   {0:<9}     {1:>25}\n"
        out_file.write(fmt_.format(*["Time", time.strftime("%H:%M:%S")]))

        fmt_ = "   {:<9} " + "    {:>25}\n\n"
        duration = int((datetime.now() - start).total_seconds())
        out_file.write(fmt_.format(*["Duration", duration]))

        fmt_ = "   {0:>9}     {1:>25}\n"
        out_file.write(fmt_.format(*["Criterion", char_floats(fval)[0]]))

        out_file.write("\n")
        fmt_ = "{:>13}    " + "{:>25}    " * 3
        out_file.write(fmt_.format(*["Identifier", "Start", "Step", "Current"]))
        out_file.write("\n\n")

        # Formatting for the file
        fmt_ = "   {:>10}" + "    {:>25}" * 3
        for i in range(num_paras):
            if paras_fixed[i]:
                continue
            line = [i] + char_floats(x_optim_container[i, :])
            out_file.write(fmt_.format(*line).rstrip(" ") + "\n")
        out_file.write("\n")

        # Get information on the spectral condition number of the covariance
        # matrix of the shock distribution.
        cond = []
        for i in range(3):
            shocks_cholesky = distribute_parameters(
                x_econ_container[:, i], paras_type="econ"
            )["shocks_cholesky"]
            shocks_cov = shocks_cholesky.dot(shocks_cholesky.T)
            cond += [np.log(_spectral_condition_number(shocks_cov))]
        fmt_ = "   {:>9} " + "    {:25.15f}" * 3 + "\n"
        out_file.write(fmt_.format(*["Condition"] + cond))

        out_file.write("\n")

        # Record warnings
        value_current = opt_obj.crit_vals[2]
        value_start = opt_obj.crit_vals[0]

        is_large = [False, False, False]
        is_large[0] = abs(value_start) > LARGE_FLOAT
        is_large[1] = abs(opt_obj.crit_vals[1]) > LARGE_FLOAT
        is_large[2] = abs(value_current) > LARGE_FLOAT

        for i in range(3):
            if is_large[i]:
                record_warning(i + 1)

    write_est_info(
        opt_obj.crit_vals[0],
        x_econ_container[:, 0],
        opt_obj.num_step,
        opt_obj.crit_vals[1],
        x_econ_container[:, 1],
        opt_obj.num_eval,
        opt_obj.crit_vals[2],
        x_econ_container[:, 2],
        num_paras,
    )


def write_est_info(
    value_start,
    paras_start,
    num_step,
    value_step,
    paras_step,
    num_eval,
    value_current,
    paras_current,
    num_paras,
):

    # Formatting for the file
    fmt_ = "{:>25}    " * 4

    # Write information to file.
    with open("est.respy.info", "w") as out_file:
        # Write out information about criterion function
        out_file.write("\n{:>25}\n\n".format("Criterion Function"))
        out_file.write(fmt_.format(*["", "Start", "Step", "Current"]) + "\n\n")

        line = [""] + char_floats([value_start, value_step, value_current])
        out_file.write(fmt_.format(*line) + "\n\n")

        out_file.write("\n{:>25}\n\n".format("Economic Parameters"))
        line = ["Identifier", "Start", "Step", "Current"]
        out_file.write(fmt_.format(*line) + "\n\n")
        for i, _ in enumerate(range(num_paras)):
            line = [i]
            line += char_floats([paras_start[i], paras_step[i]])
            line += char_floats(paras_current[i])
            out_file.write(fmt_.format(*line) + "\n")

        fmt_ = "\n{0:<25}    {1:>25}\n"
        out_file.write(fmt_.format(*[" Number of Steps", num_step]))
        out_file.write(fmt_.format(*[" Number of Evaluations", num_eval]))


def char_floats(floats):
    """Pretty printing of floats.

    Args:
        floats (list or float): if float it gets converted to a list with one
            entry.

    Returns:
        line (list): list of strings.

    """
    if isinstance(floats, float):
        floats = [floats]

    line = []
    for value in floats:
        if abs(value) > LARGE_FLOAT:
            line += ["{:>25}".format("---")]
        else:
            line += ["{:25.15f}".format(value)]

    return line


def record_estimation_final(success, message):
    """Summarize the results of the estimation."""
    with open("est.respy.log", "a") as out_file:
        out_file.write(" ESTIMATION REPORT\n\n")
        out_file.write("   Success " + str(success) + "\n")
        out_file.write("   Message " + str(message) + "\n")


def _spectral_condition_number(mat):

    svs = scipy.linalg.svd(mat)[1]
    cond = np.amax(abs(svs)) / np.amin(abs(svs))

    return cond
