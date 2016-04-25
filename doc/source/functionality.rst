Functionality
=============

There are three key objects when interacting with the **respy** package.

.. class:: respy.RespyCls(fname)

    Class for the management of the user's initialization file.

    :param str fname: Path to initialization file

.. function:: respy.simulate(respy_obj)

    Simulate dataset of synthetic agents following the model specified in the
    initialization file.

    :param obj respy_obj: Instance of RespyCls class.
    :return: Instance of RespyCls with additional information.

.. function:: respy.estimate(respy_obj)

    Estimate a model based on a provided dataset.

    :param obj respy_obj: Instance of RespyCls class.
    :return: Instance of RespyCls with additional information.
