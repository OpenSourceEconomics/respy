Introduction
==============

.. only non technical aspects
.. for what can respy be used? -> teaching, research, start for extension
.. for what has respy been used?

``respy`` is a research tool. It provides the computational support for several research projects that analyze the economics driving agents' educational and occupational choices over their life cycle within the framework of a finite-horizon discrete choice dynamic programming model.

Here is some of the recent work:

* Eisenhauer, P. (2016). `The Approximate Solution of Finite-Horizon Discrete Choice Dynamic Programming Models: Revisiting Keane & Wolpin (1994) <https://github.com/structRecomputation/manuscript/blob/master/eisenhauer.2016.pdf>`_. *Unpublished Manuscript*.

    The estimation of finite-horizon discrete choice dynamic programming models is computationally expensive. This limits their realism and impedes verification and validation efforts. Keane & Wolpin (1994) propose an interpolation method that ameliorates the computational burden but introduces approximation error. I describe their approach in detail, successfully recompute their original quality diagnostics, and provide some additional insights that underscore the trade-off between computation time and the accuracy of estimation results.

* Eisenhauer, P. (2016). Risk and Ambiguity in Dynamic Models of Educational Choice. *Unpublished Manuscript*.

    I instill a fear of model misspecification into the agents of a finite-horizon discrete choice dynamic programming model. Agents are ambiguity averse and seek robust decisions for a variety of alternative models. I study the implications for agents’ decisions and the design and impact of alternative policies.

We provide the package and its documentation to ensure the recomputability, transparency, and extensibility of this research. We also hope to showcase how software engineering practices can help in achieving these goals.

The rest of this documentation is structured as follows. First, we provide the installation instructions. Then we present the underlying economic model and discuss its solution and estimation. Next, we illustrate the basic capabilities of the package in a tutorial. We continue by providing more details regarding the numerical components of the package and showcase the package's reliability and scalability. Finally, we outline the software engineering practices adopted for the ongoing development of the package.
