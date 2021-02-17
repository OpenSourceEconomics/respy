.. implementation_kw94:

Model in Keane and Wolpin (1994)
================================

The explanations section of this documentation gives a detailed outline
of the economic modeling components and the mathematical framework in
Eckstein-Keane-Wolpin models using the example of Keane and Wolpin
(1997, :cite:`Keane.1997`). In the documentation, you will often another model
specification rooted in the publication Keane and Wolpin
(1994, :cite:`Keane.1994`). This model constitutes a similar but simpler version
of the model. We give a brief overview of the reward functions and components
distinctive to this specification here. Note that the underlying economic and
mathematical framework remains the same. 

--------------------------------------------------------------------------------

.. raw:: html

   <div
    <p class="d-flex flex-row gs-torefguide">
        <span class="badge badge-info">To Explanations</span></p>
    <p>Find the economic model and mathematical framework in the <a href="index.html">
       Explanations</a> </p>
   </div>

--------------------------------------------------------------------------------
 
The model from Keane and Wolpin (1994, :cite:`Keane.1994`) is characterized by four
distinct choices. At each point in time :math:`t \in \{0, ...,39\}` individuals decide
between :math:`a \in \{1,2,3,4\}` mutually exclusive alternatives:
working in occupation *A*, working in occupation *B* ($a=1,2$), investing in
*education* ($a=3$), or staying *home* ($a=4$). The alternatives are associated
with the rewards:

.. math::

   \text{Occupation A: } R_1(t) &= w_{1t} = r_{1}exp\{\alpha_{1} + \beta_{1,1}h_{t}
        + \gamma_{1,1}k_{1t} + \gamma_{1,2}k^2_{1t} + \gamma_{1,7}k_{2t}
        + \gamma_{2,8}k^2_{2t} + \epsilon_{1t}\} \nonumber \\
   \text{Occupation B: } R_2(t) &= w_{2t} = 
        r_{2}exp\{\alpha_{2} + \beta_{2,1}h_{t} + \gamma_{2,1}k_{2t} 
        + \gamma_{2,2}k^2_{2t} + \gamma_{2,7}k_{1t} + \gamma_{2,8}k^2_{1t} 
        + \epsilon_{2t}\} \nonumber \\
   \text{School: }R_3(t) &= \alpha_3 + \beta_{tc}I(h_t \geq 12) 
        + \beta_{rc}(1-d_3(t-1)) + \epsilon_{3t}, \nonumber \\
   \text{Home: }R_4(t) &= \alpha_4 + \epsilon_{4t}


These rewards enter the alternative specific value functions of individuals. In these 
equations :math:`h(t)` denotes schooling in period :math:`t` and :math:`k_{at}` denotes
work experience from sector :math:`A` or :math:`B` (:math:`a=1,2`). The reward for
schooling includes an indicator :math:`I(h_t \geq 12)` which is connected to the cost of
schooling after 12 periods (i.e. post-secondary schooling costs) and component that
captures costs of returning to school when the choice in the previous period was
something else. Aside from the parameters connected to these various components, each
reward function also contains a constant and an alternative specific shock. The skill
price in occupations is denoted by :math:`r_{a}`, it is set to 1 in this model. [#]_


The model from Keane and Wolpin (1994, :cite:`Keane.1994`) is not a complete
subset of the model outlined in the explanations. The most important deviations are:

    - The model includes an additional squared experience term with parameter
      :math:`\gamma_{2,8}` for experience in the other occupation.
    
    - It also does not include unobserved heterogeneity i.e. types. Here we thus
      define :math:`\alpha_{a}` as the constant for alternative :math:`a`.
      
    - We do not distinguish between different levels of post-secondary education.
      The parameters :math:`\beta_{tc}` and :math:`\beta_{tr}` are thus not enumerated.


.. rubric:: Footnotes

.. [#] Note that the reward functions are not only time but also individual specific.
       A subscript for individuals is left out for simplicity.
