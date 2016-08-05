Background
==========

The ``respy`` package is a research tool. It provides the computational support for several research papers that analyze the economics driving agents' educational and occupational choices over their life-cycle within the framework of a finite-horizon discrete choice dynamic programming model.

Here is some of the recent work:

* Eisenhauer, P. (2016). `The Approximate Solution of Finite-Horizon Discrete Choice Dynamic Programming Models: Revisiting Keane & Wolpin (1994) <https://github.com/structRecomputation/manuscript/blob/master/eisenhauer.2016.pdf>`_. Unpublished Manuscript

    The capabilities of the ``respy`` package nests the original model used in Keane and Wolpin (1994) to analyze their proposed approximation scheme. I describe their approach in detail, recompute their original quality diagnostics, and conduct some additional analysis.

* Eisenhauer, P. and Wild, S. M. (2016). *Numerical Upgrade to Finite-Horizon Discrete Choice Programming Models*. Unpublished Manuscript

    We revisit the key computational components involved in estimating finite-horizon discrete choice dynamic programming models and show how improved numerical ingredients can improve the reliability and robustness of research results.

* Eisenhauer, P. (2016). *Risk and Ambiguity in Dynamic Models of Educational Choice*. Unpublished Manuscript

    We instill a fear of model misspecification into the agents of a dynamic discrete choice model. Agents are ambiguity averse and seek robust decisions for a variety of alternative models. We study the implications for agents’ decisions and the design and impact of alternative policies.

We provide the package and its documentation to ensure the transparency, reliability, recomputability, and extensibility of this research. We hope to showcase how software engineering practices can help in achieving these objectives.

The rest of this documentation is structured as follows. First, we provide the installation instructions. Then we turn to a discussion of the underlying economic model, before we illustrate the basic capabilities of the package in a tutorial. Next, we provide more details regarding the numerical methods used for the computations. Then we showcase the package's reliability and scalability. We conclude with a presentation of the software engineering practices adopted for the ongoing development of the package.
