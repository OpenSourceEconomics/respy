Scalability
===========

The solution and estimation of finite-horizon discrete choice dynamic programming model appears straightforward. However, it entails a considerable computational burden.

During an estimation thousands of different candidate parameterizations of the model are appraised with respect to the sample likelihood. Each time we need to evaluate the four-dimensional integral of :math:`E\max` at a total of 163,410 states. The figure below illustrates the well known curse of dimensionality (Bellman, 1957) as the number of states increases exponentially with each period.

.. image:: images/state_space.png
    :width: 500px
    :align: center
    :height: 500px

To construct the sample likelihood, we also need to simulate the choice probabilities for each agent in each period to construct the sample likelihood.

Thus, in addition to Python, we also maintain a scalar and parallel Fortran implementation. We parallelize the workload using the master-slave paradigm. More precisely, we assign each slave a subset of states to evaluate the :math:`E\max` and a subset of agents to simulate their choice probabilities.

The figure below shows the total computation time required for 1,000 evaluations of the criterion function as we increase the number of slave processors. Judging against the linear benchmark, the code scales well over this range of processors.

.. image:: images/scalability.respy.png
    :width: 500px
    :align: center
    :height: 500px

Adding even more processors does not lead to any further improvements, it even increases the computational time. Adding even more slaves increases the communication time due to the required synchronization of the :math:`E\max` across all processes. Even though each slave is only working on a subset of states each period at a given time, they need access all :math:`E\max` results during the backward induction procedure.

For more details, see the script `online <https://github.com/restudToolbox/package/blob/master/development/testing/scalability/run.py>`_ and the `logfile <https://github.com/restudToolbox/package/blob/master/doc/results/scalability.respy.info>`_.
