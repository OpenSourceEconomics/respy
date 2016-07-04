Additional Details
==================


Output Files
------------

.. _data.respy.dat:
* **data.respy.dat**
    
    This file contains the agent choices and state experiences. The simulated dataset has the following structure.
    
    ======      ========================      
    Column      Information
    ======      ========================      
    1           agent identifier     
    2           time period     
    3           choice (1 = occupation A, 2 = occupation B, 3 = education, 4 = home)     
    4           earnings (missing value if not working)     
    5           work experience in occupation A     
    6           work experience in occupation B     
    7           years of schooling     
    8           lagged schooling     
    ======      ========================

.. _data.respy.paras:
* **data.respy.info**


API Reference
-------------

The API reference provides detailed descriptions of **respy** classes and
functions. It should be helpful if you plan to extend **respy** with custom components.

.. class:: respy.RespyCls(fname)

    Class to process and manage the user's initialization file.

    :param str fname: Path to initialization file
    :return: Instance of RespyCls

    .. py:classmethod:: update_model_paras(x)

        Function to update model parameterization.

        :param numpy.ndarray x: Model parameterization

.. function:: respy.simulate(respy_obj)

    Simulate dataset of synthetic agents following the model specified in the
    initialization file.

    :param obj respy_obj: Instance of RespyCls class 
    :return: Instance of RespyCls 

.. function:: respy.estimate(respy_obj)

    Estimate a model based on a provided dataset and the model specified in the initialization file.

    :param obj respy_obj: Instance of RespyCls class 

    :return: Model parameterization at final step
    :rtype: numpy.ndarray

    :return: Value of criterion function at final step
    :rtype: float


Suggested Citation
------------------

We appreciate citations for **respy** because it lets us find out how people have been using the library and it motivates further work. Please use the following sample to cite your x.y version of **respy**.

.. code-block:: text

    @misc{respy-x.y,
      title = {respy x.y},
      author = {{The respy Team}},
      year = {2016},
      howpublished = {\href{http://respy.readthedocs.io}{http://respy.readthedocs.io}},
    }

If you are unsure about which version of **respy** you are using run:

.. code-block:: bash

   $ pip show respy