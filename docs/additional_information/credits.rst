.. _credits:

Contact and Credits
===================

If you have any questions or comments, please do not hesitate to contact us via filing
an issue on Github, writing an `email`_ to our `zulipchat group
<http://gettsim.zulipchat.com/>`_ or via personal emails.

.. _email: respy.9b46528f81292a712fa4855ff362f40f.show-sender@streams.zulipchat.com


Development Lead
----------------

`Tobias Raabe <https://github.com/tobiasraabe>`_


Maintainer / Core Developer
---------------------------

- `Philipp Eisenhauer <https://github.com/peisenha>`_
- `Janos Gabler <https://github.com/janosg>`_


Acknowledgments
---------------

We are grateful to the `Social Science Computing Services <https://sscs.uchicago.edu/>`_
at the `University of Chicago <https://www.uchicago.edu/>`_ who let us use the Acropolis
cluster for scalability and performance testing. We appreciate the financial support of
the `AXA Research Fund <https://www.axa-research.org/>`_ and the  `University of Bonn
<https://www.uni-bonn.de>`_. We are indebted to the open source community as we build on
top of numerous open source tools such as the `SciPy <https://www.scipy.org>`_ and
`PyData <https://pydata.org/>`_ ecosystems.


.. Keep following section in sync with README.rst.

Citation
--------

With the second release and a complete rewrite ``respy`` evolved into a general
framework for the estimation of Eckstein-Keane-Wolpin models. Please cite it with

.. code-block::

    @Unpublished{Raabe2020,
      Title  = {respy - A Framework for the Simulation and Estimation of
                Eckstein-Keane-Wolpin Models.},
      Author = {Tobias Raabe},
      Year   = {2020},
      Url    = {https://github.com/OpenSourceEconomics/respy},
    }

Before that, ``respy`` was developed by Philipp Eisenhauer and provided a package for
the simulation and estimation of a prototypical finite-horizon discrete choice dynamic
programming model. At the heart of this release is a Fortran implementation with Python
bindings which uses MPI and OMP to scale up to HPC clusters. It is accompanied by a
Python implementation for teaching purposes. If you use ``respy`` up to version 1.2.1,
please cite it with

.. code-block::

    @Software{Eisenhauer2019,
      Title  = {respy - A Package for the Simulation and Estimation of a prototypical
                finite-horizon Discrete Choice Dynamic Programming Model.},
      Author = {Philipp Eisenhauer},
      Year   = {2019},
      DOI    = {10.5281/zenodo.3011343},
      Url    = {https://doi.org/10.5281/zenodo.3011343}
    }

We appreciate citations for ``respy`` because it helps us to find out how people have
been using the package and it motivates further work.
