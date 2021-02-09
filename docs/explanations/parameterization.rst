.. _parameterization:

Parameterization
================

.. raw:: html

   <div
    <p class="d-flex flex-row gs-torefguide">
        <span class="badge badge-info">To Explanation</span></p>
    <p>The following table keeps track of the parameterization for the computational
       implementation introduced in<a href="computational_implementation.html">
       Computational Implementation</a> </p>
   </div>

The wildcard {civilian} means either "blue" or "white".

.. csv-table:: Table of Parameterization
   :header: "Parameter", "State variable in **respy**", "Explanation"
   :widths: 20, 35, 45

   ":math:`\delta`", "delta", "discount factor"
   ":math:`e_{1,a}`", "type_1", "deviation for type 1 from type 0 in a"
   ":math:`e_{2,a}`", "type_2", "deviation for type 2 from type 0 in a"
   ":math:`e_{3,a}`", "type_3", "deviation for type 3 from type 0 in a"
   "", "**Common parameters**", ""
   ":math:`\alpha_a`", "constant", "log of rental price if the base skill endowment of type 0 is normalized to 0 (wage)"
   ":math:`\vartheta_1`", "common_hs_graduate", "common return to high school degree (non pecuniary)"
   ":math:`\vartheta_2`", "common_co_graduate", "common return to college degree (non pecuniary)"
   ":math:`\vartheta_3`", "common_hs_graduate", "effect of leaving the military early (after one year)"
   "", "**Schooling-related**", ""
   ":math:`\beta_{a,1}`", "exp_school", "linear return to an additional year of schooling (wage)"
   ":math:`\beta_{a,2}`", "exp_school", "skill premium of having finished high school (wage)"
   ":math:`\beta_{a,3}`", "exp_school", "skill premium of having finished college (wage)"
   ":math:`\beta_{tc_1}`", "hs_graduate", "net tuition costs college (non pecuniary)"
   ":math:`\beta_{tc_2}`", "co_graduate", "additional tuition costs graduate school (non pecuniary)"
   ":math:`\beta_{rc_1}`", "returns_to_high_school", "reward for going back to high school"
   ":math:`\beta_{rc_2}`", "returns_to_college", "reward for going back to college"
   "", "**Experience-related**", ""
   ":math:`\gamma_{a,1}`", "exp_{civilian}_collar", "return to experience, same sector, linear (wage)"
   ":math:`\gamma_{a,2}`", "exp_{civilian}_collar_square", "return to experience, same sector, quadratic (divided by 100) (wage)"
   ":math:`\gamma_{a,3}`", "any_exp_{civilian}_collar", "return for any experience in same sector"
   ":math:`\gamma_{a,4}`", "period", "linear age effect (wage)"
   ":math:`\gamma_{a,5}`", "is_minor", "effect of being a minor (wage)"
   ":math:`\gamma_{a,6}`", "work_{civilian}_collar_lagged", "effect of being a minor (wage)"
   ":math:`\gamma_{a,7}`", "exp_{civilian}_collar", "return to experience, other civilian sector, linear (wage)"
   ":math:`\gamma_{3,1}`", "exp_military", "return to experience, same sector, linear (wage)"
   ":math:`\gamma_{3,2}`", "exp_military_square", "return to experience, same sector, quadratic (divided by 100) (wage)"
   ":math:`\gamma_{3,3}`", "any_exp_military", "return to having any military experience"
   ":math:`\gamma_{3,4}`", "period", "linear age effect"
   ":math:`\gamma_{3,5}`", "is_minor", "effect of being a minor"
   ":math:`\gamma_{4,4}`", "period", "linear age effect"
   ":math:`\gamma_{4,5}`", "is_minor", "effect of being a minor"
   ":math:`\gamma_{5,4}`", "is_young_adult", "additional value of staying home if aged 18-20"
   ":math:`\gamma_{5,5}`", "is_adult", "additional value of staying home if 21 or older"
   "", "**Mobility and search**", ""
   ":math:`c_{a,1}`", "not_exp_{civilian}_collar_lagged", "reward of switching to a from other occupation (non pecuniary)"
   ":math:`c_{a,2}`", "not_any_exp_{civilian}_collar", "reward of working in a for the first time (non pecuniary)"
   ":math:`c_{3,2}`", "not_any_exp_military", "reward of being in the military sector for the first time (non pecuniary)"
