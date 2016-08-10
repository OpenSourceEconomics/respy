Scalability
===========

The solution and estimation a the dynamic programming model by a backward induction procedure appears straightforward. However, it entails a considerable computational burden.

Consider the evaluation of the four-dimensional integral for the :math:`E\max` as an example during an estimation, where thousands of different candidate parameterizations of the model are appraised with respect to the sample likelihood. Each time, even in the simplified models considered in Keane (1994), we need to evaluate the :math:`E\max` at a total of 163,410 states. The figure below illustrates the well known curse of dimensionality (Bellman, 1957) as the number of states increases exponentially with each period.

.. image:: images/state_space.png
    :width: 500px
    :align: center
    :height: 500px

To construct the sample likelihood, we also need to simulate the choice probabilities for each agent in each period to construct the sample likelihood.

Thus, in addition to our Python implementation, we also maintain a scalar and parallel Fortran implementation. We parallelize the workload using the master-slave paradigm. More precisely, we assign each slave a subset of states to evaluate the :math:`E\max` and a subset of agents to simulate their choice probabilities.

The figure below shows the total computation time required for 1,000 evaluations of the criterion function as the number of slave processors increases. Judging against the linear benchmark, the code scales well over this range of processors.

.. image:: images/scalability.respy.png
    :width: 500px
    :align: center
    :height: 500px

Adding even more processors does not lead to any further improvements, it even increases the computational time. Adding additional slaves has the cost of increasing communication time due to the required synchronization of the :math:`E\max` results each period. Even though each processor is only working on a subset of states each period at a given time, they need access all previous results during the backward induction procedure.

For more details, see the script `online <https://github.com/restudToolbox/package/blob/master/development/testing/scalability/run.py>`_ and the `logfile <https://github.com/restudToolbox/package/blob/master/doc/results/scalability.respy.info>`_.
