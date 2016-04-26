Major Functionality
===================

You really just need to know about three key elements of the package to started.

.. class:: respy.RespyCls(fname)

    Class for the management of the user's initialization file.

    :param str fname: Path to initialization file
    :return: Instance of RespyCls

    .. py:classmethod:: update_model_paras(x)

        Function to update model parameterization.

        :param numpy.ndarray x: Model parameterization

.. function:: respy.simulate(respy_obj)

    Simulate dataset of synthetic agents following the model specified in the
    initialization file.

    :param obj respy_obj: Instance of RespyCls class.
    :return: Instance of RespyCls 

.. function:: respy.estimate(respy_obj)

    Estimate a model based on a provided dataset.

    :param obj respy_obj: Instance of RespyCls class.

    :return: Model parameterization at final step
    :rtype: numpy.ndarray

    :return: Value of criterion function at final step
    :rtype: float