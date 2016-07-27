Software Engineering
====================

We now briefly discuss our software engineering practices that help to ensure the transparency, reliability, and extensibility of the **respy** package.

Development Infrastructure
--------------------------

We maintain a dedicated development and testing servers on the `Amazon Elastic Compute Cloud (EC2) <https://aws.amazon.com/ec2/>`_. We treat our infrastructure as code which is thus versionable, testable, and repeatable. We create the machine images using `Packer <https://www.packer.io/>`_ and manage the compute resources with `Terraform <https://www.terraform.io/>`_. Our definition files are available `online <https://github.com/restudToolbox/package/tree/master/tools>`_.

Program Design
--------------

We learned a lot about the program design from the codes provided by the original authors (`original codes <https://github.com/restudToolbox/package/tree/master/forensics>`_). Our own source code is available on `GitHub <https://github.com/restudToolbox/package>`_ as well. We maintain a pure *Python* implementation with a focus on readability and a pure *Fortran* implementation to adress any performance constraints. The user can specify the implementation in the *PROGRAM* section of the initialization file. Also, we support the use of multiple cores with the **FORTRAN** implementation.

Test Battery
------------

We build a customized testing infrastructure for the **respy** package. Among others, We rely on tools such as the the continuous integration services provided by `Travis CI <https://travis-ci.org/restudToolbox/package>`_, py.test for the management of the automated testing, and tox to ensure the property working of the package for alternative Python workings.

We broadly group our testing efforts in four categories:

* **property testing**

    We hit the program with random requests and test valid return.

* **regression testing**

    We have a set of 1,000 fixed model parameterizations and the corresponding estimation results. Any changes to the code are tested against these results.

* **scalability testing**

    The computational burden of estimating meaningful models quite large due to the curse of dimensionality. That motivates the parallelized version of the program. We regularly test the scalability of our code against the linear benchmark.

* **reliability testing**

    We conduct several Monte Carlo exercises to ensure that we can recover the true underlying parameterization. This also allows us to get a feeling for the what parameters are easily recovered and which are not. We document one of these Monte Carlo exercises in the earlier section.

* **release testing**

    Often new releases only consist of refactoring or cleanup of the code base. So, the results should actually be unaffected by the changes. So, as part of finalizing a new release, we test the new release against the results from the new release for random requests. The same is true, if subset of user requests should be unaffected by the changes.

Documentation
-------------

The documentation is build with `Sphinx <http://www.sphinx-doc.org/>`_. Hosted on readthedocs.io

Code Review
-----------

We use several automatic code review tools to ensure the readability and maintainability of our code base. For example, we rely on `Quantified Code <https://www.quantifiedcode.com/app/project/b00436d2ca614437b843c7042dba0c26>`_ and `Codacy <https://www.codacy.com/app/eisenhauer/respy/dashboard>`_.

Continuous Integration Workflow
-------------------------------

We set up a continuous integration workflow using `Travis CI <https://travis-ci.org/restudToolbox/package>`_. We use `tox <https://tox.readthedocs.io>`_ to ensure the correct installation of the package with different *Python* environments and automate our builds using `Waf <https://waf.io/>`_. We rely on `GitHub <https://github.com/restudToolbox/package>`_ as our online version control system and follow the `Gitflow Workflow <https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow>`_. We use Gitlab for issue tracking.

Release Management
------------------

We distribute our software through `PyPI <https://pypi.python.org/pypi/respy>`_ which automatically updated from Travis-CI. We do so by first pushing the new release to `TestPyPI <https://testpypi.python.org/pypi>`_ and running the test battery for a couple of days.
