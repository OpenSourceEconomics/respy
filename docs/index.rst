.. Keep the following section in sync with README.rst.

respy
=====

.. image:: https://anaconda.org/opensourceeconomics/respy/badges/version.svg
    :target: https://anaconda.org/OpenSourceEconomics/respy

.. image:: https://anaconda.org/opensourceeconomics/respy/badges/platforms.svg
    :target: https://anaconda.org/OpenSourceEconomics/respy

.. image:: https://readthedocs.org/projects/respy/badge/?version=latest
    :target: https://respy.readthedocs.io/en/latest

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT

.. image:: https://github.com/OpenSourceEconomics/respy/workflows/Continuous%20Integration%20Workflow/badge.svg?branch=master
    :target: https://github.com/OpenSourceEconomics/respy/actions?query=branch%3Amaster

.. image:: https://codecov.io/gh/OpenSourceEconomics/respy/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/OpenSourceEconomics/respy

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

----

**respy** is an open source framework written in Python for the simulation and
estimation of some :term:`finite-horizon` :term:`discrete choice` :term:`dynamic
programming` models.

With conda available on your path, installing **respy** is as simple as typing

.. code-block:: bash

    $ conda config --add channels conda-forge
    $ conda install -c opensourceeconomics respy

The documentation is structured in four parts.

.. raw:: html

    <div class="container" id="index-container">
        <div class="row">
            <div class="col-lg-3 col-md-6 col-sm-6 col-xs-12 d-flex">
                <a href="tutorials/index.html" id="index-link">
                    <div class="card text-center intro-card shadow">
                        <img src="_static/images/light-bulb.svg" class="card-img-top"
                             alt="tutorials-icon" height="52"
                        >
                        <div class="card-body flex-fill">
                            <h5 class="card-title">Tutorials</h5>
                            <p class="card-text">
                                Tutorials help you to get started. They cover the basics
                                and are designed for everyone new to the package.
                            </p>
                        </div>
                    </div>
                </a>
            </div>
            <div class="col-lg-3 col-md-6 col-sm-6 col-xs-12 d-flex">
                <a href="how_to_guides/index.html" id="index-link">
                    <div class="card text-center intro-card shadow">
                        <img src="_static/images/book.svg" class="card-img-top"
                             alt="how-to guides icon" height="52"
                        >
                        <div class="card-body flex-fill">
                            <h5 class="card-title">How-to Guides</h5>
                            <p class="card-text">
                                How-to guides are designed to provide detailed
                                instructions for very specific and advanced tasks.
                            </p>
                        </div>
                    </div>
                </a>
            </div>
            <div class="col-lg-3 col-md-6 col-sm-6 col-xs-12 d-flex">
                <a href="explanations/index.html" id="index-link">
                    <div class="card text-center intro-card shadow">
                        <img src="_static/images/books.svg" class="card-img-top"
                             alt="explanations icon" height="52"
                        >
                        <div class="card-body flex-fill">
                            <h5 class="card-title">Explanations</h5>
                            <p class="card-text">
                                Explanations give detailed information to key topics and
                                concepts which underlie the package.
                            </p>
                        </div>
                    </div>
                </a>
            </div>
            <div class="col-lg-3 col-md-6 col-sm-6 col-xs-12 d-flex">
                <a href="reference_guides/index.html" id="index-link">
                    <div class="card text-center intro-card shadow">
                        <img src="_static/images/coding.svg" class="card-img-top"
                             alt="reference guides icon" height="52"
                        >
                        <div class="card-body flex-fill">
                            <h5 class="card-title">Reference Guides</h5>
                            <p class="card-text">
                                Reference Guides explain the implementation. If you are
                                interested in the inner workings, you will find this
                                section helpful.
                            </p>
                        </div>
                    </div>
                 </a>
            </div>
        </div>
    </div>


.. toctree::
    :hidden:

    tutorials/index
    explanations/index
    how_to_guides/index
    reference_guides/index


If you are looking for other resources, you might find them here.

.. toctree::
    :maxdepth: 1

    api
    glossary
    about_us
    projects
    development/index
    replications/index
    changes
