Software Engineering
====================

We now briefly discuss our software engineering practices that help us to ensure the
transparency, reliability, scalability, and extensibility of the ``respy`` package.

Development Infrastructure
--------------------------
.. todo:: Do you still use Chef? I am confused since there is no definition file
          availabe in the `tools_dir <https://github.com/OpenSourceEconomics/respy/
          tree/janosg/tools>`_.

We maintain a dedicated development and testing server on the `Amazon Elastic Compute
Cloud <https://aws.amazon.com/ec2/>`_. We treat our infrastructure as code thus making
it versionable, testable, and repeatable. We create our machine images using `Packer
<https://www.packer.io/>`_ and `Chef <https://www.chef.io/>`_ and manage our compute
resources with `Terraform <https://www.terraform.io/>`_. Our definition files are
available `here <https://github.com/OpenSourceEconomics/respy/tree/janosg/tools>`_.

Program Design
--------------

We build on the design of the original authors (`codes <https://github.com/
OpenSourceEconomics/respy/tree/janosg/development/documentation/forensics>`_). We
maintain a pure Python implementation with a focus on readability and a scalar and
parallel Fortran implementation to address any performance constraints. We keep the
structure of the Python and Fortran implementation aligned as much as possible. For
example, we standardize the naming and interface design of the routines across versions.

Test Battery
------------

.. image:: https://codecov.io/gh/restudToolbox/package/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/restudToolbox/package

We use `pytest <http://docs.pytest.org>`_ as our test runner. We broadly group our tests
in four categories:

* **property-based testing**

    We create random model parameterizations and estimation requests and test for a
    valid return of the program. For example, we estimate the same model specification
    using the parallel and scalar implementations as both results need to be identical.
    Also, we maintain a an ``f2py`` interface to ensure that core functions of our
    Python and Fortran implementation return the same results. Finally, we also upgraded
    the codes by Keane and Wolpin (1994) and can compare the results of the ``respy``
    package with their implementation for a restricted set of estimation requests that
    are valid for both programs.

* **regression testing**

    We retain a set of 10,000 fixed model parameterizations and store their estimation
    results. This allows to ensure that a simple refactoring of the code or the addition
    of new features does not have any unintended consequences on the existing
    capabilities of the package.

* **scalability testing**

    We maintain a scalar and parallel Fortran implementation of the package, we
    regularly test the scalability of our code against the linear benchmark.

* **reliability testing**

    We conduct numerous Monte Carlo exercises to ensure that we can recover the true
    underlying parameterization with an estimation. Also by varying the tuning
    parameters of the estimation (e.g. random draws for integration) and the optimizers,
    we learn about their effect on estimation performance.

* **release testing**

    We thoroughly test new release candidates against previous releases. For minor and
    micro releases, user requests should yield identical results. For major releases,
    our goal is to ensure that the same is true for at least a subset of requests. If
    required, we will build supporting code infrastructure.

* **robustness testing**

    Numerical instabilities often only become apparent on real world data that is less
    well behaved than simulated data. To test the stability of our package we start
    thousands of estimation tasks on the NLSY dataset used by Keane and Wolpin. We use
    random start values for the parameter vector that can be far from the true values
    and make sure that the code can handle those cases.

Our `tests <https://github.com/OpenSourceEconomics/respy/tree/janosg/respy/tests>`_ and
the `testing infrastructure <https://github.com/OpenSourceEconomics/respy/tree/janosg/
development/testing>`_ are available online. As new features are added and the code
matures, we constantly expand our testing harness. We run a test battery nightly on our
development server, see `here <https://github.com/OpenSourceEconomics/respy/blob/master/
example/ec2-respy.testing.log>`__ for an example output.

Documentation
-------------
.. todo:: All batches reefer to the no longer existing restudtoolbox repository.

.. image:: https://readthedocs.org/projects/respy/badge/?version=latest
   :target: https://respy.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

The documentation is created using `Sphinx <http://www.sphinx-doc.org/>`_ and hosted on `Read the Docs <https://readthedocs.org/>`_.

Code Review
-----------

.. image:: https://api.codacy.com/project/badge/Grade/3dd368fb739c49d78d910676c9264a81
   :target: https://www.codacy.com/app/eisenhauer/respy?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=restudToolbox/package&amp;utm_campaign=Badge_Grade

.. image:: https://landscape.io/github/restudToolbox/package/master/landscape.svg?style=flat
    :target: https://landscape.io/github/restudToolbox/package/master
    :alt: Code Health

We use several automatic code review tools to help us improve the readability and
maintainability of our code base. For example, we work with `Codacy
<https://app.codacy.com/app/eisenhauer/respy/dashboard>`_ and `Landscape
<https://landscape.io/github/restudToolbox/package>`_

Continuous Integration Workflow
-------------------------------

.. image:: https://api.travis-ci.org/OpenSourceEconomics/respy.svg?branch=master
   :target: https://travis-ci.org/OpenSourceEconomics/respy

We set up a continuous integration workflow around our `GitHub Organization
<https://github.com/OpenSourceEconomics>`_. We use the continuous integration services
provided by `Travis CI <https://travis-ci.org/restudToolbox/package>`_. `tox
<https://tox.readthedocs.io>`_ helps us to ensure the proper workings of the package for
alternative Python implementations. Our build process is managed by `Waf
<https://waf.io/>`_. We rely on `Git <https://git-scm.com/>`_ as our version control
system and follow the `Gitflow Workflow
<https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow>`_. We use
`Github <https://github.com/OpenSourceEconomics/respy/issues>`_ for our issue tracking.
The package is distributed through `PyPI <https://pypi.org/project/respy/>`_ which
automatically updated from our development server.
