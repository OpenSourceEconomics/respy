Software Engineering
====================

We now briefly discuss our software engineering practices that help us to ensure the transparency, reliability, and extensibility of the ``respy`` package.

Development Infrastructure
--------------------------

We maintain a dedicated development and testing server on the `Amazon Elastic Compute Cloud (EC2) <https://aws.amazon.com/ec2/>`_. We treat our infrastructure as code thus making it versionable, testable, and repeatable. We create our machine images using `Packer <https://www.packer.io/>`_ and `Chef <https://www.chef.io/>`_ and manage our compute resources with `Terraform <https://www.terraform.io/>`_. Our definition files are available `online <https://github.com/restudToolbox/package/tree/master/tools>`_.

Program Design
--------------

We build on the design of the original authors (`codes <https://github.com/restudToolbox/package/tree/master/forensics>`_). We maintain a pure Python implementation with a focus on readability and a scalar and parallel Fortran implementation to address any performance constraints. We keep the structure of the Python and Fortran implementation aligned as much as possible. This includes the naming and interface design of the subroutines and functions for example.

Test Battery
------------

.. image:: https://codecov.io/gh/restudToolbox/package/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/restudToolbox/package

We use `pytest <http://docs.pytest.org>`_ as our test runner. We broadly group our tests in four categories:

* **property testing**

    We create random model parameterizations and estimation requests and test for a valid return of the program. Among many other tests, we estimate the same model specification using the parallel and scalar implementations as both results need to be identical. Also, we maintain a an ``f2py`` interface to ensure that core functions of our Python and Fortran implementation return the same results. Finally, we even upgraded the codes by the original authors and can compare the results of the ``respy`` package with their implementation for a restricted set of estimation requests that are valid for both programs.

* **regression testing**

    We maintain a set of 1,000 fixed model parameterizations and store their estimation results. This allows to ensure that a simple refactoring of the code or the addition of new features does not have any unintended consequences on the existing capabilities of the package.

* **scalability testing**

    As we maintain a scalar and parallel Fortran version of the package, we regularly test the scalability of our code against the linear benchmark.

* **reliability testing**

    We conduct numerous Monte Carlo exercises to ensure that we can recover the true underlying parameterization with an estimation. Also by varying the tuning parameters of the estimation (e.g. random draws for integration) and the optimizers, we learn about their effect on estimation performance.

* **release testing**

    New release candidates are thoroughly tested against the previous release. In most cases, the results for at least as subset of model specifications and estimation requests should be identical.

Our `tests <https://github.com/restudToolbox/package/tree/master/respy/tests>`_ and the `testing infrastructure <https://github.com/restudToolbox/package/tree/master/development/testing>`_ are available online. We run a test battery nightly on our development server, see `here <https://github.com/restudToolbox/package/blob/master/example/ec2-respy.testing.log>`_  for an example output.

Documentation
-------------

.. image:: https://readthedocs.org/projects/respy/badge/?version=latest
   :target: http://respy.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

The documentation is hosted on `Read the Docs <https://readthedocs.org/>`_ and created using `Sphinx <http://www.sphinx-doc.org/>`_.

Code Review
-----------

.. image:: https://www.quantifiedcode.com/api/v1/project/b00436d2ca614437b843c7042dba0c26/badge.svg
   :target: https://www.quantifiedcode.com/app/project/b00436d2ca614437b843c7042dba0c26
   :alt: Code issues

.. image:: https://api.codacy.com/project/badge/Grade/3dd368fb739c49d78d910676c9264a81
   :target: https://www.codacy.com/app/eisenhauer/respy?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=restudToolbox/package&amp;utm_campaign=Badge_Grade

.. image:: https://landscape.io/github/restudToolbox/package/master/landscape.svg?style=flat
    :target: https://landscape.io/github/restudToolbox/package/master
    :alt: Code Health

We use several automatic code review tools to help us improve the readability and maintainability of our code base. For example, we rely on `Quantified Code <https://www.quantifiedcode.com/app/project/b00436d2ca614437b843c7042dba0c26>`_, `Codacy <https://www.codacy.com/app/eisenhauer/respy/dashboard>`_, and `Landscape <https://landscape.io/github/restudToolbox/package>`_

Continuous Integration Workflow
-------------------------------

.. image:: https://travis-ci.org/restudToolbox/package.svg?branch=master
   :target: https://travis-ci.org/restudToolbox/package

.. image:: https://requires.io/github/restudToolbox/package/requirements.svg?branch=master
    :target: https://requires.io/github/restudToolbox/package/requirements/?branch=master
    :alt: Requirements Status

.. image:: https://badge.fury.io/py/respy.svg
    :target: https://badge.fury.io/py/respy

We set up a continuous integration workflow around our `GitHub Organization <https://github.com/restudToolbox>`_. We use the continuous integration services provided by `Travis CI <https://travis-ci.org/restudToolbox/package>`_. `tox <https://tox.readthedocs.io>`_ helps us to ensure the proper workings of the package for alternative Python implementations. Our build process is managed by `Waf <https://waf.io/>`_. We rely on `Git <https://git-scm.com/>`_ as our version control system and follow the `Gitflow Workflow <https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow>`_. We use `GitLab <https://gitlab.com/restudToolbox/package>`_ for our issue tracking. The package is distributed through `PyPI <https://pypi.python.org/pypi/respy>`_ which automatically updated from our development server.
