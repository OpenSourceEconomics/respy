Introduction
==============

``respy`` is a research tool. It provides the computational support for several research
projects that analyze the economics driving agents' educational and occupational choices
over their life cycle within the framework of a finite-horizon discrete choice dynamic
programming model.

Here is some of the recent work:

* Eisenhauer, P. (2019). `The Approximate Solution of Finite-Horizon Discrete Choice
  Dynamic Programming Models: Revisiting Keane & Wolpin (1994)
  <https://doi.org/10.1002/jae.2648>`_. *Journal of Applied Econometrics, 34* (1),
  149-154.

      The estimation of finite-horizon discrete choice dynamic programming models is
      computationally expensive. This limits their realism and impedes verification and
      validation efforts. Keane & Wolpin (1994) propose an interpolation method that
      ameliorates the computational burden but introduces approximation error. I
      describe their approach in detail, successfully recompute their original quality
      diagnostics, and provide some additional insights that underscore the trade-off
      between computation time and the accuracy of estimation results.

* Eisenhauer, P. (2018). `Robust human capital investment under risk and ambiguity
  <https://github.com/peisenha/peisenha.github.io/blob/master/material/
  eisenhauer-robust.pdf>`_. *Revise and resubmit at the Journal of Econometrics*.

      I instill a fear of model misspecification into the agents of a finite-horizon
      discrete choice dynamic programming model. Agents are ambiguity averse and seek
      robust decisions for a variety of alternative models. I study the implications for
      agents’ decisions and the design and impact of alternative policies.

We provide the package and its documentation to ensure the recomputability,
transparency, and extensibility of this research. We also hope to showcase how software
engineering practices can help in achieving these goals.
