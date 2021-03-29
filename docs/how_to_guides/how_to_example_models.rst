How to Load Example Models
==========================

In the tutorials and guides of the **respy** documentation, you will frequently 
encounter *example models*. These are pre-defined models that can be easily accessed
with just one function call to facilitate an introductory workflow and use in
explanatory material. The set of examples consists of simple as well as very advanced
models to cover a wide range of use cases and challenges that come with an increasing
degree of complexity in dynamic discrete choice models.

For instance, in the tutorials, you will encounter two very simple models that are based 
on the story of Robinson Crusoe. Other example models are based on actual economic
publications and thus serve to illustrate the scalability of `respy` models.

You can access example models by typing 

.. code-block:: python

    params, options, data = rp.get_example_model("model_name")


If you only want to get the `params` and `options`, set the argument `with_data`
to **False**.

.. code-block:: python

    params, options = rp.get_example_model("model_name", with_data=False)



.. raw:: html

     <div
      <p class="d-flex flex-row gs-torefguide">
          <span class="badge badge-info">To API</span></p>
      <p>For more information, checkout the function in the<a
         href="../autoapi/respy/interface/index.html#respy.interface.get_example_model">API</a> </p>
     </div>


Below are the example models that are currently available. 

-----

.. tabs::

   .. tab:: Toy Models

      The package provides users with two toy models. These models can be used to acquaintance 
      oneself with **respy**'s functionalities and can be encountered in the tutorials.

      .. raw:: html

           <div
            <p class="d-flex flex-row gs-torefguide">
                <span class="badge badge-info">To Tutorials</span></p>
            <p>Check out the <a
               href="../tutorials/index.html">Tutorials</a> </p>
        </div>


      The models are called:

      - ``robinson_crusoe_basic``
      - ``robinson_crusoe_extended``

      These models are excellent examples to use for learning and prototyping: They
      include a small number of available choices and a low number of periods such
      that they are computationally feasible.
      

   .. tab:: KW (1994)

      Aside from toy models, **respy** also provides several models that stem from the
      economic literature on dynamic life-cycle models. The most simple examples are a group
      of models based on the following publication:

      - Keane, M. P., & Wolpin, K. I. (1994). The Solution and Estimation of Discrete Choice
      Dynamic Programming Models by Simulation and Interpolation: Monte Carlo Evidence.
      *The Review of Economics and Statistics*, 648-672.

      In the study, Keane and Wolpin (1994) develop an approximate solution method which
      consists of Monte Carlo integration with simulation and an interpolation approach to
      ease the computational burden of solving the DCDP model. They utilize one model with
      three different parametrizations to assess their solution method. This model and its
      three parametrizations are used as example models in the **respy** interface.
      
      They are called:
      
      - ``kw_94_one``
      - ``kw_94_two``
      - ``kw_94_three``
      
      The model consists of four mutually exclusive alternatives that individuals can choose
      in each period. Agents can either choose to work in one of two sectors *a* or *b*,
      invest in *education* or stay *home*. The work alternatives award a wage and experience,
      while school only awards experience. In the home option, individuals gain neither a wage
      nor experience. The plot below shows the choice patterns for the three parametrizations.
      The model considers a time horizon of 40 periods.

   .. tab:: KW (1997)

      A more advanced group of examples are given by the models developed by Keane and Wolpin
      (1997). In this study, the authors implement an empirical structural life-cycle model of
      occupational choice and human capital investment. They estimate their model on data
      from the National Longitudinal Survey of Youth (NLSY). The study includes a "basic"
      model parametrization that is very similar to the model of Keane and Wolpin (1994) and
      and "extended" parametrization that improves on the empirical fit of the basic model.

      - Keane, M. P., & Wolpin, K. I. (1997). The Career Decisions of Young Men. 
      *Journal of Political Economy*, 105(3), 473-522.
      
      **respy** supports both the basic and extended parametrization from the paper.
      They are named:

      - ``kw_97_basic``
      - ``kw_97_extended``
      
      However, the parametrization from the paper returns different life-cycle patterns for
      **respy** than presented in the paper. You can thus also access our estimates based for
      the models that are based on the same empirical data by adding ``_respy`` to the model
      name. 
      
      The models consist of three occupational choices (*white collar*, *blue collar*, and 
      *military*), one educational choice (*school*), and a *home* option. Both models
      consider a life-cycle of 50 periods. These models are decidedly larger than the toy
      models and require a considerable amount of computation power to solve, especially the
      extended model.

   .. tab:: KW (2000)
   
      Another example model provided in the respy interface is the model presented in Keane
      and Wolpin (2000). The model incorporates an observable characteristic to account for
      race, aiming to analyze the effects of monetary incentive schemes designed to reduce
      racial differences in school attainment and earnings.

      - Keane, M. P., & Wolpin, K. I. (2000). Eliminating Race Differences in School
        Attainment and Labor Market Success. *Journal of Labor Economics*, 18(4), 614-652.

      The model is named 

      - ``kw_2000``

      The model is very similar to the extended model specification in Keane and Wolpin
      (1997).
