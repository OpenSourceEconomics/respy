Reliability
===========

We document th results of two basic Monte Carlo exercises to illustrate the reliability of the **respy** package. Again, we use the first parameterization from Keane & Wolpin (1994) as our `baseline model specification <https://github.com/restudToolbox/package/blob/master/respy/tests/resources/kw_data_one.ini>`_ and simulate a sample of 1,000 agents.  Then we run two estimations with alternative starting values and evaluate the root-mean squared error (RMSE) of the choice probabilities. Throughout, we allow for 3,000 evaluations of the criterion function.

... starting at true values
---------------------------

We start at the true parameters. While taking a total of 1,491 steps, the actual effect on the parameter values and the criterion function is negligible. The RMSE remains literally unchanged at 0.00.

    =====   ====    =====   ===========
    Start   Stop    Steps   Evaluations
    =====   ====    =====   ===========
    0.00    0.00    1,491   3,000
    =====   ====    =====   ===========

... starting with myopic agents
-------------------------------

To create a more challenging exercise, we create a set of starting values by first estimating a static model, i.e. :math:`\delta = 0` on the simulated dataset. The results from this misspecified estimation run serve as the starting values for a then correctly specified estimation. For the *static* estimation, we start with a RMSE of about 0.44 which, after 950 steps is cut to 0.25. Most of this discrepancy is driven by relatively low educational investment whose perceived returns are low to the myopic agents.

    =====   ====    =====   ===========
    Start   Stop    Steps   Evaluations
    =====   ====    =====   ===========
    0.44    0.25     950    3,000
    =====   ====    =====   ===========

Using the resulting values, we start with a RMSE of about 0.23 which is then reduced to only 0.01 after 1,453 steps of the optimizer.

    =====   ====    =====   ===========
    Start   Stop    Steps   Evaluations
    =====   ====    =====   ===========
    0.23    0.01    1,453   3,000
    =====   ====    =====   ===========

For more details, see the script `online <https://github.com/restudToolbox/package/blob/master/development/testing/reliability/run.py>`_. The results for all the parameterizations analyzed in Keane & Wolpin (1994) are available `here <https://github.com/restudToolbox/package/blob/master/development/testing/reliability/reliability.respy.base>`_. Overall the results are encouraging. However, doubts about the correctness of our implementation always remain. So, if you are struggling with a particularly poor performance in your application, please do not hesitate to let us know so we can help with the investigation.

Note that in all those estimations we solve the complete dynamic programming problems and do not rely on the interpolation approach proposed by Keane & Wolpin (1994). For an assessment of the quality of the approximation method see the analysis in Eisenhauer (2016).
