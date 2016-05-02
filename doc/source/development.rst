Development Setup
=================

Program Design
--------------

We have learned a lot from the codes of the original codes. You can check them out `online <https://github.com/restudToolbox/package/tree/master/forensics>`_, where we also provide upgraded versions of the key modules. Our own source code is available on `GitHub <https://github.com/restudToolbox/package>`_ as well.

The **respy** package contains three alternative implementations for each of the key steps of model, i.e. solution, simulation, and estimation. We maintain a pure *Python* implementation with a focus on readability and a pure *Fortran* implementation to evade any performance constraints. In addition, we also provide a *F2PY* implementation, where key functions are written in *Fortran* and called from *Python* to improve performance. However, the main purpose of the *F2PY* implementation is to ease the development process by allowing to incrementally write *FORTRAN* subroutines and test them within a *Python* testing framework. The user can specify the implementation in the *PROGRAM* section of the initialization file. 

Test Battery
------------

You can have a look at our tests `online <https://github.com/restudToolbox/package/tree/master/respy/tests>`_. We also build a testing infrastructure, that allows to test the functionality of the **respy** package for numerous alternative parameterizations (`link <https://github.com/restudToolbox/package/tree/master/development/testing>`_). We also upgraded the original *Fortran* codes developed by the authors and compare simulated samples between our package and their implementation. Our testing process is automated using **pytest**.

Continuous Integration Workflow
--------------------------------

We have set up a continuous integration workflow using `Travis CI <https://travis-ci.org/restudToolbox/package>`_. We use `tox <https://tox.readthedocs.io>`_ to ensure the a correct package installation with different *Python* environments and automate our builds using `Waf <https://waf.io/>`_. We are rely `GitHub <https://github.com/restudToolbox/package>`_ as our online version control system and distribute our software through `PyPI <https://pypi.python.org/pypi/respy>`_. The documentation is build with `Sphinx <http://www.sphinx-doc.org/>`_.