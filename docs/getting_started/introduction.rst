Introduction
==============

``respy`` is a tool to solve, simulate, and estimate structural econometric models.
It provides the computational support for several research projects that analyze
the economics driving agents' educational and occupational choices over their life
cycle within the framework of a finite-horizon discrete choice dynamic
programming model.

While written in pure Python, it's speed can rival Fortran implementations of
similar models. This is due to a heavy use of `numba <http://numba.pydata.org/>`_,
a just in time compiler for Python.

``respy`` is under ongoing development. We add new features every week and try to
make it more flexible and easier to use without sacrificing execution speed.

The goal is to make it possible to solve, simulate and estimate all models that
have been classified as the Keane-Wolpin-Eckstein branch of structural econometrics
in a `survey <https://www.sciencedirect.com/science/article/pii/S0304407609001985>`_
by Aguirregabiria & Mira (2010).

This includes but is not limited to the following Models:

- Keane, M. P. & Wopin, W. I. `(1997) <https://www.jstor.org/stable/10.1086/262080>`_.
    The Career Decisions of Young Men, Journal of Political Economy, 105(3): 473-552.

- Keane, M. P. & Wopin, W. I. `(1994) <https://www.jstor.org/stable/2109768>`_.
    The Solution and Estimation of Discrete Choice Dynamic Programming Models by
    Simulation and Interpolation: Monte Carlo Evidence, The Review of Economics and
    Statistics, 76(4): 648-672.

... and we are almost there!
 
If you have any questions or comments, please do not hesitate to contact us via  `SLACKXXX <https://github.com/OpenSourceEconomics/respy>`_ or by filing an issue on `GitHub <https://github.com/OpenSourceEconomics/respy>`_ or .
