Scalability
===========

We allow for parallel computing using `MPICH <https://www.mpich.org/>`_, an implementation of the `MPI <http://www.mpi-forum.org/>`_ standard. We exploit the multiple processors using the master-slave paradigm. We distribute the calculation of the expected future values and the sample likelihood across multiple processors. For the latter, each period, we split up the total number of states across the available cores. For the former, we simple assign the a subset of agents to each core.


.. image:: images/scalability.respy.png