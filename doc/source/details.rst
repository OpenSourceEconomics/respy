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
    3           choice, 1 = work A, 2 = work B, 3 = education, 4 = home)     
    4           earnings (missing value if not working)     
    5           work experience in occupation A     
    6           work experience in occupation B     
    7           years of schooling     
    8           lagged schooling     
    ======      ========================

.. _data.respy.paras:
* **data.respy.paras**

    This file contains the coefficients used for the simulation of the agent
    population.

    =======     ========================      
    Lines       Coefficients
    =======     ========================       
    1 -  6      occupation A      
    7 - 12      occupation B     
    12 - 15     education     
    16          home     
    16 - 26     cholesky factors     
    =======     ========================

    Note, that the last ten coefficients do refer to the Cholesky factors of the
    covariance matrix of the shocks and not the covariance matrix itself. 


.. _paras.respy.log:
* Parameters logging

    * **paras_curre.respy.log**, current candidate parameters

    * **paras_start.respy.log**, parameters at the start of the optimization

    * **paras_steps.respy.log**, parameters at last step of the optimizer
