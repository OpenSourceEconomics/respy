Software Engineering
====================

We now briefly discuss our software engineering practices that help to ensure the transparency, reliability, and extensibility of the ``respy`` package.

Development Infrastructure
--------------------------

We maintain a dedicated development and testing server on the `Amazon Elastic Compute Cloud (EC2) <https://aws.amazon.com/ec2/>`_. We treat our infrastructure as code thus making it versionable, testable, and repeatable. We create our machine images using `Packer <https://www.packer.io/>`_ and `Chef <https://www.chef.io/>`_ and manage our compute resources with `Terraform <https://www.terraform.io/>`_. Our definition files are available `online <https://github.com/restudToolbox/package/tree/master/tools>`_.

Program Design
--------------

We build on the design of the original authors (`codes <https://github.com/restudToolbox/package/tree/master/forensics>`_). We maintain a pure PYTHON implementation with a focus on readability and a scalar and parallel FORTRAN implementation to address any performance constraints. We keep the structure of the Python and Fortran implementation aligned as much as possible.

Test Battery
------------

We use `pytest <http://docs.pytest.org>`_ for our automated testing efforts. We broadly group our testing efforts in four categories:

* **property testing**

    We hit the program with random requests and test valid return. For example, we estimate the same model specification using the parallel and scalar implementations. The results need to be identical. By maintaining an ``f2py`` interface we test core functionality between our Python and Fortran implementations. We upgraded the codes by the original authors and can compare the results of the ``respy`` package with their for a restricted set of estimation requests.
    
* **regression testing**

    We maintain a set of 1,000 fixed model parameterizations and store their estimation results. This allows to ensure that a simple refactoring of the code or the addition of features do not have any unintended consequences on the existing capabilities of the package.

* **scalability testing**

    The computational burden of estimating meaningful models quite large due to the curse of dimensionality. That motivates the parallelized version of the program. We regularly test the scalability of our code against the linear benchmark.

* **reliability testing**

    We conduct numerous Monte Carlo exercises to ensure that we can recover the true underlying parameterization with an estimation. This allows to develop a better understanding for what parameters are easily recovered and which are not. Also by varying the tuning parameters of the estimation (e.g. smoothing parameters) and the optimizers we learn about their effect on estimation performance.

* **release testing**

    New release candidates are thoroughly tested against the previous release. In most cases, the results for at least as subset of model specifications should be identical.

Our `tests <https://github.com/restudToolbox/package/tree/master/respy/tests>`_ and the `testing infrastructure <https://github.com/restudToolbox/package/tree/master/development/testing>`_ are available online.

Documentation
-------------

The documentation is hosted on `Read the Docs <https://readthedocs.org/>`_ and created using `Sphinx <http://www.sphinx-doc.org/>`_.

Code Review
-----------

We use several automatic code review tools to ensure the readability and maintainability of our code base. For example, we rely on `Quantified Code <https://www.quantifiedcode.com/app/project/b00436d2ca614437b843c7042dba0c26>`_ and `Codacy <https://www.codacy.com/app/eisenhauer/respy/dashboard>`_.

Continuous Integration Workflow
-------------------------------

We set up a continuous integration workflow around our `GitHub Organization <https://github.com/restudToolbox>`_. We use continuous integration services provided by `Travis CI <https://travis-ci.org/restudToolbox/package>`_ and `tox <https://tox.readthedocs.io>`_ to ensure the proper workings of the package for alternative Python implementations. Our build process is managed by `Waf <https://waf.io/>`_. We rely on `Git <https://git-scm.com/>`_ as our version control system and follow the `Gitflow Workflow <https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow>`_. We use `GitLab <https://gitlab.com/restudToolbox/package>`_ for our issue tracking. The package is distributed through `PyPI <https://pypi.python.org/pypi/respy>`_ which automatically updated from our development server.
