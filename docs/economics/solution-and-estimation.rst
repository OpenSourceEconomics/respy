Solution and Estimation
=========================

Solution
--------

From a mathematical perspective, this type of model boils down to a finite-horizon DP
problem under uncertainty that can be solved by backward induction. For the discussion,
it is useful to define the value function :math:`V(S(t),t)` as a shorthand for the
agents objective function. :math:`V(S(t),t)` depends on the state space at :math:`t` and
on :math:`t` itself due to the finiteness of the time horizon and can be written as:

.. math::

    V(S(t),t) = \max_{k \in K}\{V_k(S(t),t)\},

with :math:`V_k(S(t),t)` as the alternative-specific value function. :math:`V_k(S(t),t)`
obeys the Bellman equation (Bellman, 1957) and is thus amenable to a backward recursion.

.. math::

    \begin{align}
        V_k(S(t),t) = \begin{cases}
            R_k(S(t)) + \delta E\left[V(S(t + 1), t + 1) \mid
                S(t), d_k(t) = 1\right] &\text{if } t < T \\
            R_k(S(t)) &\text{if } t = T.
        \end{cases}
    \end{align}

Assuming continued optimal behavior, the expected future value of state :math:`S(t + 1)`
for all :math:`K` alternatives given today's state :math:`S(t)` and choice :math:`d_k(t)
= 1`, :math:`E\max_k(S(t + 1))` for short, can be calculated:

.. math::

    E\max_k(S(t + 1)) = E\left[V(S(t + 1), t + 1) \mid| S(t), d_k(t) = 1\right].

This requires the evaluation of a :math:`K` - dimensional integral as future rewards are
partly uncertain due to the unknown realization of the shocks:

.. math::

  \begin{align}
      E\max(S(t)) &= \\
        \int_{\epsilon_1(t)} ... \int_{\epsilon_K(t)}\max\{R_1(t), ..., R_K(t)\}f_{\epsilon}(\epsilon_1(t), ... ,\epsilon_K(t))d\epsilon_1(t) ... d\epsilon_K(t),
      E\max(S(t)) &= \int_{\epsilon_1(t)} \dots \int_{\epsilon_K(t)} \max\{ \\
                &R_1(t) + \delta E\max_1(S(t+1)), \\
                &\dots, \\
                &R_K(t) + \delta E\max_K(S(t+1)) \\
                \} f_\epsilon(\epsilon_1(t), \dots , \epsilon_K(t)) d\epsilon_1(t) \dots d\epsilon_K(t)
    \end{align}

where :math:`f_{\epsilon}` is the joint density of the uncertain component of the
rewards in :math:`t` not known at :math:`t - 1`. With all ingredients at hand, the
solution of the model by backward induction is straightforward.

Estimation
----------

We estimate the parameters of the reward functions :math:`\theta` based on a sample of
agents whose behavior and state experiences are described by the model. Although all
shocks to the rewards are eventually known to the agent, they remain unobserved by the
econometrician. So each parameterization induces a different probability distribution
over the sequence of observed agent choices and their state experience. We implement
maximum likelihood estimation and appraise each candidate parameterization of the model
using the likelihood function of the observed sample (Fisher, 1922). Given the serial
independence of the shocks, We can compute the likelihood contribution by agent and
period. The sample likelihood is then just the product of the likelihood contributions
over all agents and time periods. As we need to simulate the agent's choice
probabilities, we end up with a simulated maximum likelihood estimator (Manski and
Lerman, 1977) and minimize the simulated negative log-likelihood of the observed sample.
