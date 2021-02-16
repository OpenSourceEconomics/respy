.. _how_to_guides:

How-to Guides
=============

How-to guides provide detailed information about **respy**'s functionalities.
Refer to these guides if you aim to implement and estimate your own EKW model.

.. raw:: html

  <header>
    <h2>Model implementation</h2>
  </header>
    <div class="container" id="index-container">
        <div class="row">
            <div class="col-lg-4 col-md-4 col-sm-4 col-xs-12 d-flex">
                <a href="specify_model.html" id="index-link">
                    <div class="card text-center shadow h-100">
                        <div class="card-header"><large>How to Specify a Model</large></div>
                            <div class="card-body flex-fill">
                                <p class="card-text small">
                                    Learn how to translate a model from equations
                                    to the params and options.
                            </p>
                        </div>
                    </div>
                </a>
            </div>

            <div class="col-lg-4 col-md-4 col-sm-4 col-xs-12 d-flex">
                <a href="example_models.html" id="index-link">
                    <div class="card text-center shadow h-100">
                        <div class="card-header"><large>Loading Example Models</large></div>
                            <div class="card-body flex-fill">
                                <p class="card-text small">
                                    Access pre-defined toy examples and published
                                    models. 
                            </p>
                        </div>
                    </div>
                </a>
            </div>

            <div class="col-lg-4 col-md-4 col-sm-4 col-xs-12 d-flex">
                <a href="simulation.html" id="index-link">
                    <div class="card text-center shadow h-100">
                        <div class="card-header"><large>Simulation</large></div>
                            <div class="card-body flex-fill">
                                <p class="card-text small">
                                    This guide contains a comprehensive overview
                                    of simulation capabilities.
                            </p>
                        </div>
                    </div>
                </a>
            </div>
        </div>
    </div>




  <header>
    <h2>Model extensions</h2>
  </header>
    <div class="container" id="index-container">
        <div class="row">
            <div class="col-lg-4 col-md-4 col-sm-4 col-xs-12 d-flex">
                <a href="finite_mixture.html" id="index-link">
                    <div class="card text-center shadow h-100">
                        <div class="card-header"><large>Unobeserved Hetereogeneity</large></div>
                            <div class="card-body flex-fill">
                                <p class="card-text small">
                                    Learn how to implement finite mixture models
                                    to account for unobserved heterogeneity.
                            </p>
                        </div>
                    </div>
                </a>
            </div>

            <div class="col-lg-4 col-md-4 col-sm-4 col-xs-12 d-flex">
                <a href="initial_conditions.html" id="index-link">
                    <div class="card text-center shadow h-100">
                        <div class="card-header"><large>Initial Conditions</large></div>
                            <div class="card-body flex-fill">
                                <p class="card-text small">
                                    Add observable heterogeneity like experience,
                                    observable characteritics, and lagged choices
                                    to your model.
                            </p>
                        </div>
                    </div>
                </a>
            </div>

            <div class="col-lg-4 col-md-4 col-sm-4 col-xs-12 d-flex">
                <a href="simulation.html" id="index-link">
                    <div class="card text-center shadow h-100">
                        <div class="card-header"><large>Impatient Robinson</large></div>
                            <div class="card-body flex-fill">
                                <p class="card-text small">
                                    Learn how to implement hyperbolic discounting
                                    in your model.
                            </p>
                        </div>
                    </div>
                </a>
            </div>
        </div>
    </div>



  <header>
    <h2>Model Calibration</h2>
  </header>
    <div class="container" id="index-container">
        <div class="row">
            <div class="col-lg-4 col-md-4 col-sm-4 col-xs-12 d-flex">
                <a href="likelihood.html" id="index-link">
                    <div class="card text-center shadow h-100">
                        <div class="card-header"><large>Maximum Likelihood</large></div>
                            <div class="card-body flex-fill">
                                <p class="card-text small small">
                                    Learn how to construct a criterion function for maximum
                                    likelihood estimation.
                            </p>
                        </div>
                    </div>
                </a>
            </div>

            <div class="col-lg-4 col-md-4 col-sm-4 col-xs-12 d-flex">
                <a href="msm.html" id="index-link">
                    <div class="card text-center shadow h-100">
                        <div class="card-header"><large>Method of Simulated Moments</large></div>
                            <div class="card-body flex-fill">
                                <p class="card-text small">
                                    Construct a criterion function for estimation using
                                    the method of simulated moments.
                            </p>
                        </div>
                    </div>
                </a>
            </div>

            <div class="col-lg-4 col-md-4 col-sm-4 col-xs-12 d-flex">
                <a href="simulation.html" id="index-link">
                    <div class="card text-center shadow h-100">
                        <div class="card-header"><large>Estimation Exercise MSM</large></div>
                            <div class="card-body flex-fill">
                                <p class="card-text small">
                                    This guide contains an example of method of simulated
                                    moments estimation using estimagic.
                            </p>
                        </div>
                    </div>
                </a>
            </div>
        </div>
    </div>


.. toctree::
    :hidden:
    :maxdepth: 1

    specify_model
    example_models
    simulation
    finite_mixture
    initial_conditions
    impatient_robinson
    numerical_integration
    likelihood
    msm
    msm_estimation_exercise
