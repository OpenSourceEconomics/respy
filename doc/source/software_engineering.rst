Software Engineering
====================

In this section, we provide some background about the tools from software engineering that allow to code a reliable, well designed and tested implementation.

Program Design
--------------

We learned a lot about the program design from the codes provided by the original authors (`original codes <https://github.com/restudToolbox/package/tree/master/forensics>`_). Our own source code is available on `GitHub <https://github.com/restudToolbox/package>`_ as well.

The **respy** package contains two alternative implementations for each of the two key steps of the package, e.g. simulation and estimation. We maintain a pure *Python* implementation with a focus on readability and a pure *Fortran* implementation to evade any performance constraints. The user can specify the implementation in the *PROGRAM* section of the initialization file. Also, we support the use of multiple cores with the **FORTRAN** implementation. See our section on Scalability for more details on the issue.


Code Quality
------------

Code Review
~~~~~~~~~~~

We use several automatic code review tools to ensure the readability and maintainability of our code base. For example, we rely on `Quantified Code <https://www.quantifiedcode.com/app/project/b00436d2ca614437b843c7042dba0c26>`_ and `Codacy <https://www.codacy.com/app/eisenhauer/respy/dashboard>`_. 

Test Battery
~~~~~~~~~~~~

You can have a look at our tests `online <https://github.com/restudToolbox/package/tree/master/respy/tests>`_. We also build a testing infrastructure, that allows to test the functionality of the **respy** package for numerous alternative parameterizations (`link <https://github.com/restudToolbox/package/tree/master/development/testing>`_). This property-based testing turned out very powerful, as during the estimation the optimizer may visit quite extreme parameterizations of the model. Also, as the **respy** design is settling down, we have numerous regression tests implemented that ensure that simple refactoring does not entail any unintended consequences. We also upgraded the original *Fortran* codes developed by the authors and compare simulated samples between our package and their implementation. We automate our test execution using **pytest**. We have set up a testing server on the `Amazon Elastic Compute Cloud (EC2) <https://aws.amazon.com/ec2/>`_ for daily runs of our whole test battery. We maintain interfaces to selected **FORTRAN** functions using **F2PY** which allows us to directly test for the equality of **PYTHON** and **FORTRAN** functions. However, the main purpose of the *F2PY* implementation is to ease the development process by allowing to incrementally write *FORTRAN* subroutines and test them within a *Python* testing framework. 

Continuous Integration Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We set up a continuous integration workflow using `Travis CI <https://travis-ci.org/restudToolbox/package>`_. We use `tox <https://tox.readthedocs.io>`_ to ensure the correct installation of the package with different *Python* environments and automate our builds using `Waf <https://waf.io/>`_. We rely on `GitHub <https://github.com/restudToolbox/package>`_ as our online version control system and follow the `Gitflow Workflow <https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow>`_. 

Release Management
------------------

We distribute our software through `PyPI <https://pypi.python.org/pypi/respy>`_ which automatically updated from `Travis CI <https://travis-ci.org/restudToolbox/package>`_. Each new release of the **respy** package is extensively tested. Given the stability of the package by now, each new release usually contains only new features which allow to regression tests against the previous release for requests contained in both. We do so by first pushing the new release to `TestPyPI <https://testpypi.python.org/pypi>`_. The documentation is build with `Sphinx <http://www.sphinx-doc.org/>`_. And then ...


