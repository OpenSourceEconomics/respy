Tutorial 
========

As a user of the **respy** package, we usually either want to simulate a synthetic sample form the 
model or start an estimation run. At the heart of our interaction with the **respy** package is 
the *RobupyCls*. This classes processes the user's model specification. We can initialize an instance of the class by passing in the location of the model initialization file.
::

    from respy import RespyCls

    respy_obj = RespyCls('data_one.ini')

This then allows us to simulate a sample from the model::

    from respy import simulate

    simulate('data_one.ini')

The sample is simulated with the parameters specified in the initialization file, which are discussed in
Section :ref:`specification`.

During the simulation, several files will appear in the current working
directory.

* **logging.respy.sol.log**

    This file contains logging information from the solution algorithm

* **logging.respy.sim.log**

    This file contains logging information from the simulation algorithm. It is
    updated as the simulation progresses and also reports the random seed for
    the agent state experiences.

The names of the following files depend on that is specified for the filename in
the SIMULATION section of the model initialization file.

* **data.respy.dat**
    
    This file contains the agent choices and state experiences. The dataset has
    the following structure.
    
    ======      ========================      
    Column      Information
    ======      ========================      
    1           agent identifier     
    2           time period     
    3           Choice, 1 = Work A, 2 = Work B, 3 = Education, 4 = Home)     
    4           Earnings (missing value if not working)     
    5           work experience in occupation A     
    6           work experience in occupation B     
    7           years of schooling     
    8           lagged schooling     
    ======      ========================

* **data.respy.info**

    This file provides some basic descriptive information about the simulated
    agent population such as aggregate choice probabilities and wage
    distributions.

* **data.respy.paras**

    This file contains the coefficients used for the simulation of the agent
    population.

    =======     ========================      
    Lines       Coefficients
    =======     ========================       
    1 -  6      Occupation A      
    7 - 12      Occupation B     
    12 - 15     Education     
    16          Home     
    16 - 26     Cholesky Factors     
    =======     ========================

    Note, that the last ten coefficients do refer to the Cholesky factors of the
    covariance matrix of the shocks and not the covariance matrix directly. The
    Cholesky factors are in order of a flattened upper triangular matrix.

