Reliability
===========

We document th results of two basic Monte Carlo exercises to illustrate the reliability of the **respy** package. Again, we use the first parameterization from Keane & Wolpin (1994) as our `baseline model specification <https://github.com/restudToolbox/package/blob/master/respy/tests/resources/kw_data_one.ini>`_ and simulate a sample of 1,000 agents.  Then we run an estimation and evaluate the root-mean squared error (RMSE) of the choice probabilities. Throughout, we allow for 3,000 evaluations of the criterion function.

Starting at True Values
-----------------------

First, we start at the true parameters. While taking a total of 1,340 steps, the actual effect on the parameter values and the criterion function is negligible. The RMSE remains literally unchanged at 0.00.

    =====   ====    =====   ===========
    Start   Stop    Steps   Evaluations
    =====   ====    =====   ===========
    0.00    0.00    1,321   2,000
    =====   ====    =====   ===========

Starting at Static Model
------------------------

To create a more challenging, but still well-behaved estimation challenge, we create a set of starting values by first estimating a static $\delta$, and thus misspecified, model and feeding in the converged parameters as starting values for a correctly specified estimation run. For the *static* estimation, we start with a RMSE of about 0.4 which, after 1,213 steps is cut about in half. Most of this discrepancy is driven by relatively low educational investment.

    =====   ====    =====   ===========
    Start   Stop    Steps   Evaluations
    =====   ====    =====   ===========
    0.00    0.00    1,321   2,000
    =====   ====    =====   ===========

Using the resulting parameterization as the starting value for the *dynamic*, correctly specified model, we start with a RMSE of about 0.25 which is then reduced to only 0.06 after 2,312 steps of the optimizer.

    =====   ====    =====   ===========
    Start   Stop    Steps   Evaluations
    =====   ====    =====   ===========
    0.00    0.00    1,321   2,000
    =====   ====    =====   ===========

For more details, see the script `online <https://github.com/restudToolbox/package/blob/master/development/testing/reliability/run.py>`_. The results for the all the parameterizations analyzed in Keane & Wolpin (1994) are available `here <https://github.com/restudToolbox/package/blob/master/development/testing/reliability/reliability.respy.base>`_. Overall the results are encouraging. However, doubts about the correctness of our implementation always remain. So, if you are struggling with a particularly poor performance in your application, please do not hesitate to let us know so we can help with the investigation.
