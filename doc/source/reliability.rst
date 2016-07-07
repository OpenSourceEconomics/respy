Reliability
===========

We perform document two Monte Carlo exercises here to illustrate the reliability of the **respy** package. We numerous additional Monte Carlo exercises during the development process. Overall, we are encouraged by the results of our Monte Carlo exercises and additional testing. However, doubts about the correctness of our implementation always remain. So, if you are struggling with a particularly poor performance in your application, please do not hesitate to let us know so we can help with the investigation.

Again, we use the specifications from Keane & Wolpin (1994) and simulate a sample of 1,000 agents. Then we run an estimation and evaluate the root-mean squared error (RMSE) of the choice probabilities. For more details, see the script online. 

Starting at True Values
-----------------------

First, we start at the true parameters. Allowing for 2,000 evaluations of the criterion function the optimizer takes 1,340 steps. However, the effect on parameter values and the criterion function is negligible. The RMSE remains literally unchanged at 0.00.

Starting at Misspecified Model
------------------------------

To create a more challenging, but still well-behaved estimation challenge, we create a set of starting values by first estimating a static $\delta$, and thus misspecified, model and feeding in the converged parameters as starting values for a correctly specified estimation run. For the *static* estimation, we start with a RMSE of about 0.4 which, after 1,213 steps is cut about in half. Most of this discrepancy is driven by relatively low educational investment. Using the resulting parameterization as the starting value for the *dynamic*, correctly specified model, we start with a RMSE of about 0.25 which is then reduced to only 0.06 after 2,312 steps of the optimizer.

