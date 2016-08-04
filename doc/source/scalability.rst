Scalability
===========


The evaluation of :math:`E\max` at each possible state creates a considerable computational burden. For example, even in this simplified model, it requires the repeated evaluation of the integral for the :math:`E\max` at a total of 163,410 states. During an estimation, the model has to be solved repeatedly for numerous alternative parameterizations.

.. image:: images/state_space.png

This figure imposes the restriction that agents can only obtain 10 additional years of schooling.

For the parallel implementation of the model, we distribute the total computational work across the multiple processors using the master-slave paradigm. More precisely, we distribute the approximation of the expected future values and the sample likelihood across multiple processors. For the latter, each period, we split up the total number of states across the available slaves. For the former, we simple assign the a subset of agents to each slave

Using the `baseline model specification <https://github.com/restudToolbox/package/blob/master/respy/tests/resources/kw_data_one.ini>`_, the figure below shows the total computation time required for 2,000 evaluations of the criterion function as the number of slave processors increases. Judging against the linear benchmark, the code scales well over the range of processors.

.. image:: images/scalability.respy.png

For more details, see the script `online <https://github.com/restudToolbox/package/blob/master/development/testing/scalability/run.py>`_. The results for all the parameterizations analyzed in Keane & Wolpin (1994) are available `here <https://github.com/restudToolbox/package/blob/master/development/testing/scalability/scalability.respy.base>`_.
