Reliability
===========

We document the results of two straightforward Monte Carlo exercises to illustrate the reliability of the ``respy`` package. We use the first parameterization from Keane (1994) and simulate a sample of 1,000 agents. Then we run two estimations with alternative starting values. Each time, we use the estimation results to again simulate a sample using the same random draws and evaluate the root-mean squared error (RMSE) of the simulated choice probabilities. We use the NEWUOA algorithm with its default tuning parameters and allow for a maximum of 3,000 evaluations of the criterion function.

... starting at true values
---------------------------

Initially we start at the true parameter values. While taking a total of 1,491 steps, the actual effect on the parameter values and the criterion function is negligible. The RMSE remains literally unchanged at zero.

    =====   ====    =====   ===========
    Start   Stop    Steps   Evaluations
    =====   ====    =====   ===========
    0.00    0.00    1,491   3,000
    =====   ====    =====   ===========

... starting with myopic agents
-------------------------------

Again we start from th true parameters of the reward functions, but estimate a static (:math:`\delta = 0`) model on the simulated sample. This results from this misspecified estimation then serve as the starting values for a the dynamic (:math:`\delta = 0.95`) version of the model

For the static estimation, we start with a RMSE of about 0.44 which, after 950 steps is cut to 0.25. Most of this discrepancy is driven by the relatively low school enrollments as there is no investment motive for the myopic agents.

    =====   ====    =====   ===========
    Start   Stop    Steps   Evaluations
    =====   ====    =====   ===========
    0.44    0.25     950    3,000
    =====   ====    =====   ===========

We then est up the estimation of the dynamic model using the initial results as starting values. Initially, the RMSE is about 0.23 but is quickly reduced to only 0.01 after 1,453 steps of the optimizer.

    =====   ====    =====   ===========
    Start   Stop    Steps   Evaluations
    =====   ====    =====   ===========
    0.23    0.01    1,453   3,000
    =====   ====    =====   ===========

Overall the results are encouraging. However, doubts about the correctness of our implementation always remain. So, if you are struggling with a particularly poor performance in your application, please do not hesitate to let us know so we can help with the investigation.

For more details, see the script `online <https://github.com/restudToolbox/package/blob/master/development/testing/reliability/run.py>`_. The results for all the parameterizations analyzed in Keane (1994) are available `here <https://github.com/restudToolbox/package/blob/master/doc/results/reliability.respy.info>`_. Note that in all those estimations we solve the complete dynamic programming problems and do not rely on the interpolation approach proposed by Keane (1994).
