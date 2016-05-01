Development Setup
=================

We have set up a continuous integration workflow using `Travis CI <https://travis-ci.org/restudToolbox/package>`_. We are using `GitHub <https://github.com/restudToolbox/package>`_ as our online version control system and distribute our software through `PyPI <https://pypi.python.org/pypi/respy>`_. The documentation is build with `Sphinx <http://www.sphinx-doc.org/>`_. We use `Waf <https://waf.io/>`_ as our build automation tool.

Test Battery
------------

You can have a look at our tests `online <https://github.com/restudToolbox/package/tree/master/respy/tests>`_. We also build a testing infrastructure, that allows to test the functionality of the **respy** package for numerous alternative parameterizations (`link <https://github.com/restudToolbox/package/tree/master/development/testing>`_). This setup, inspired by the idea of property based testing, has proven tremendously useful to check the validity of the code for numerous admissible parameterizations. We also upgraded the original *Fortran* codes developed by the authors and compare simulated samples between our package and their implementation. We use `tox <https://tox.readthedocs.io>`_ to ensure the a correct package installation with different *Python* environments.

