Scalability
===========

The solution and estimation of the dynamic programming model by a backward induction procedure is straightforward on a conceptual level. However, the evaluation of the integrals for the :math:`E\max` and the choice probabilities at each decision node creates a considerable computational burden. Both types of these four-dimensional integrals are approximated by Monte Carlo integration.

Let us consider again the `baseline model specification <https://github.com/restudToolbox/package/blob/master/respy/tests/resources/kw_data_one.ini>`_ from Keane and Wolpin (1994). Even in this kind of simplified model the estimation requires the repeated evaluation of the integral for the :math:`E\max` and choice probabilities at a total of 163,410 nodes. The figure below illustrates the well known curse of dimensionality (Bellman, 1957) as the number of nodes to consider increases exponentially with each period.

.. image:: images/state_space.png

This is why we provide scalar and parallel Fortran implementations. For the parallel implementation of the model, we distribute the total computational work across the multiple processors using the master-slave paradigm. More precisely, we distribute the approximation of the expected future values and the sample likelihood across multiple processors. For the latter, each period, we split up the total number of states across the available slaves. For the former, we simple assign the a subset of agents to each slave

The figure below shows the total computation time required for 1,000 evaluations of the criterion function as the number of slave processors increases. Judging against the linear benchmark, the code scales well over the range of processors.

.. image:: images/scalability.respy.png

For more details, see the script `online <https://github.com/restudToolbox/package/blob/master/development/testing/scalability/run.py>`_. The results for all the parameterizations analyzed in Keane & Wolpin (1994) are available `here <https://github.com/restudToolbox/package/blob/master/development/testing/scalability/scalability.respy.base>`_.
