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

.. image:: https://github.com/OpenSourceEconomics/respy/workflows/Continuous%20Integration%20Workflow/badge.svg?branch=main
    :target: https://github.com/OpenSourceEconomics/respy/actions?query=branch%3Amain

.. image:: https://codecov.io/gh/OpenSourceEconomics/respy/branch/main/graph/badge.svg
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

The documentation provides multiple sections:

.. raw:: html

  <header>
    <h1>Introduction</h1>
  </header>
    <div class="container" id="index-container">
        <div class="row">
            <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex">
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
            <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex">
                <a href="explanations/index.html" id="index-link">
                    <div class="card text-center intro-card shadow">
                        <img src="_static/images/books.svg" class="card-img-top"
                             alt="explanations icon" height="52"
                        >
                        <div class="card-body flex-fill">
                            <h5 class="card-title">Explanations</h5>
                            <p class="card-text">
                                Explanations contain a theoretical outline of
                                the economic models that can be implemented with
                                the package.

                            </p>
                        </div>
                    </div>
                </a>
            </div>
        </div>
    </div>
  <header>
    <h1>Advanced usage</h1>
  </header>
    <div class="container" id="index-container">
        <div class="row">
            <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex">
                <a href="how_to_guides/index.html" id="index-link">
                    <div class="card text-center intro-card shadow">
                        <img src="_static/images/book.svg" class="card-img-top"
                             alt="how-to guides icon" height="52"
                        >
                        <div class="card-body flex-fill">
                            <h5 class="card-title">How-to Guides</h5>
                            <p class="card-text">
                                How-to guides provide information on specific features and offer
                                guidance for implementing models.
                            </p>
                        </div>
                    </div>
                </a>
            </div>
            <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex">
                <a href="autoapi/index.html" id="index-link">
                    <div class="card text-center intro-card shadow">
                        <img src="_static/images/computer.svg" class="card-img-top"
                             alt="API icon" height="52"
                        >
                        <div class="card-body flex-fill">
                            <h5 class="card-title">API Reference</h5>
                            <p class="card-text">
                            The API provides a documentation of the interface and
                            individual functions in respy.
                            </p>
                        </div>
                    </div>
                 </a>
            </div>
        </div>
    </div>
  <header>
    <h1>Development</h1>
  </header>
    <div class="container" id="index-container">
        <div class="row">
            <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex">
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
            <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex">
                <a href="development/index.html" id="index-link">
                    <div class="card text-center intro-card shadow">
                        <img src="_static/images/gears.svg" class="card-img-top"
                             alt="Development icon" height="52"
                        >
                        <div class="card-body flex-fill">
                            <h5 class="card-title">Development</h5>
                            <p class="card-text">
                            There are multiple ways to contribute to respy. This
                            section documents the development of the package and outlines
                            possible contributions.
                            </p>
                        </div>
                    </div>
                 </a>
            </div>
        </div>
    </div>
  <header>
    <h1>Community </h1>
  </header>
    <div class="container" id="index-container">
        <div class="row">
            <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex">
                <a href="projects/index.html" id="index-link">
                    <div class="card text-center intro-card shadow">
                        <img src="_static/images/dividers.svg" class="card-img-top"
                             alt="projects icon" height="52"
                        >
                        <div class="card-body flex-fill">
                            <h5 class="card-title">Projects</h5>
                            <p class="card-text">
                                This section provides an overview of academic
                                projects that use respy.
                            </p>
                        </div>
                    </div>
                 </a>
            </div>
            <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex">
                <a href="about_us.html" id="index-link">
                    <div class="card text-center intro-card shadow">
                        <img src="_static/images/peoples.svg" class="card-img-top"
                             alt="about us icon" height="52"
                        >
                        <div class="card-body flex-fill">
                            <h5 class="card-title">About us</h5>
                            <p class="card-text">
                                Here you find information about the team behind the
                                development of respy.
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
    autoapi/index
    reference_guides/index
    development/index
    projects/index
    about_us
