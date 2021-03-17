Glossary
========

.. glossary::


    backward induction 
        Process of reasoning backwards in time to determine an optimal sequence of actions. The backward induction
        algorithm starts from the last period at which a decision is to be made and identifies the optimal action
        at that point in time. With this information, the process continues backwards until determining the optimal
        action at every point in time. In :term:`dynamic programming <dynamic programming>`, backwards induction
        is one of the most frequently used methods for solving the Bellman Equation. 

    bluecollar occupation
        Occupation related to manual work that might involved skilled or unskilled labor. Blue-collar occupations
        typically involve manufacturing, agriculture, construction, mining, among others.

    covariate
        A covariate always has a corresponding parameter and can be part of the rewards
        or other parts defined in ``params``. It has to be differentiated from a
        :term:`state <state space>` because a covariate is defined by some function
        :math:`g` on a :term:`state <state space>` :math:`s`.

    decision rule
        Function that maps an observation to an appropriate action.

    discounted utility
        The utility associated with a future point in time :math:`t+1` but perceived at time 
        :math:`t. Future utility is discounted back to the present using a discount factor which
        captures the notion of impatience in individual preferences.
        
    discrete choice
        A term used in economics and control theory to refer to settings in which two or
        more discrete alternatives are available.

    dynamic programming
        A mathematical optimization and a computer programming method which allows to
        break up a complex problem into smaller sub-problems in a recursive manner.

    finite-horizon
        A property of a problem or model where time usually lasts from 0 to some $T$.

    finite mixture model
        Model-based approach that accounts for the fact that regression models or distributions might differ across
        groups or subgroups of the population. A finite mixture approach then models the probability of belonging
        to each unobserved group and estimates the parameters of a regression for each group.

    forward-looking
        The assumption that economic agents form expectations of the future based on the structure of the economy
        today. Under this assumption, the announcement of a policy can alter agent’s expectations and the
        development of the economy.

    maximum likelihood estimation
        Method for estimating the parameters of a statistical model given certain observed data, by maximizing
        the likelihood function of the probability distribution of the data.
    
    measurement error
        Statistical concept defined as the difference between the measured value of a parameter and its true value.
    
    non-pecuniary utility
        Utility that is associated with non-monetary payoffs. In **respy**, we usually use the term to refer to
        components of the reward functions that are not related to wages.
    
    observable characteristic
        An observable characteristic is a trait of individuals which is exogenously 
        determined. It can either be treated as constant or malleable by exogenous
        processes.

    parameter space
        Space of possible parameter values that define a particular mathematical or statistical model.
        
    principle of optimality 
        Principle related to the :term:`dynamic programming <dynamic programming>` method and prescribes that
        no matter which is the initial state and decision of the process, the remaining decisions must constitute
        an optimal policy with regard to the state resulting from the first decision.

    sheepskin effect
        Concept that describes the phenomenon that people who posses a complete academic degree earn
        a higher income than people who do not hold a degree but posses the same amount of knowledge and skills.

    state space
        The state space is the entirety of states in a structural model. Each state in
        the state space is unique and can also be thought of as a position on a playing
        field.

        The dimensions of the state space comprise the necessary information for the
        model which can either be endogenously determined like experiences or
        exogenously like :term:`observable characteristics <observable characteristic>`. 

    stochastic shock
        Unexpected or unpredictable event that affects the economy, either positively or negatively.
        Technically, it is an unpredictable change in exogenous factors. 

    unobserved heterogeneity
        Term that describes the existence of unmeasured differences among individuals within a sample that are
        correlated with observed variables of interest. Unobserved variables in a model might lead to biased
        estimates.

    rational expectations
        Assumption that agents in a model have the ability to use all the information they have about how
        the economy operates to make their decisions.

    white-collar occupation
        Occupation related to a person who performs usually high skilled labor in an office or an
        administrative setting.
