.. _about_us:

About us
========

If you have any questions or comments, please do not hesitate to contact us via filing
an issue on Github, writing an `email`_ to or join our `zulipchat group
<https://ose.zulipchat.com/>`_ or via personal emails.

.. _email: research-codes-respy.9b46528f81292a712fa4855ff362f40f.show-sender@streams.zulipchat.com


.. tabs::

   .. tab:: Team

      The respy development team is currently a group of researchers, doctoral students, and
      students at the University of Bonn.

      **Project Manager**

      `Philipp Eisenhauer <https://github.com/peisenha>`_ (`email
      <mailto://eisenhauer@policy-lab.org>`__)

      **Software Design**

      - `Janos Gabler <https://github.com/janosg>`_ (`email
        <mailto://janos.gabler@gmail.com>`__)
      - `Tobias Raabe <https://github.com/tobiasraabe>`_ (`email 
        <mailto://raabe@posteo.de>`__)

      **Developers**

      - `Annica Gehlen <https://github.com/amageh>`_ (MSM interface)
      - `Moritz Mendel <https://github.com/mo2561057>`_ (Flexible choice sets, MSM interface)
      - `Maximilian Blesch <https://github.com/Max Blesch>`_ (Exogenous processes)


      **Contributors**

      - `Sofia Badini <https://github.com/SofiaBadini>`_ (Hyperbolic discounting)
      - `Linda Maokomatanda <https://github.com/lindamaok899>`_ (Robust OLS)
      - `Tim Mensinger <https://github.com/timmens>`_ (Recommended reading)
      - `Rafael Suchy <https://github.com/rafaelsuchy>`_ (Quasi-Monte Carlo simulation, Explanations)
      - `Benedikt Kauf <https://github.com/bekauf>`_ (Explanations)


   .. tab:: Acknowledgments

      We are grateful to the `Social Science Computing Services <https://sscs.uchicago.edu/>`_
      at the `University of Chicago <https://www.uchicago.edu/>`_ who let us use the Acropolis
      cluster for scalability and performance testing. We appreciate the financial support of
      the `AXA Research Fund <https://www.axa-research.org/>`_ and the  `University of Bonn
      <https://www.uni-bonn.de>`_.

      We gratefully acknowledge funding by the Federal Ministry of Education 
      and Research (BMBF) and the Ministry of Culture and Science of the
      State  of North Rhine-Westphalia (MKW) as part of the Excellence
      Strategy of the federal and state governments.

      We are indebted to the open source community as we build on top of numerous open source
      tools such as the `SciPy <https://www.scipy.org>`_ and `PyData <https://pydata.org/>`_
      ecosystems. In particular, without **respy**'s interface would not work without `pandas
      <https://pandas.pydata.org/>`_ and it could not rival any program written in Fortran in
      terms of speed without `Numba <http://numba.pydata.org/>`_.

      We use icons by `svgrepo.com <https://www.svgrepo.com/>`_ in the documentation.

      |OSE| |space| |TRA| |space| |UniBonn| |space| |DIW|

      .. |OSE| image:: https://raw.githubusercontent.com/OpenSourceEconomics/ose-logos/main/OSE_logo_RGB.svg
        :width: 10%
        :target: https://open-econ.org

      .. |UniBonn| image:: _static/images/UNI_Bonn_Logo_Standard_RZ_RGB.svg
        :width: 20 %
        :target: https://www.uni-bonn.de

      .. |TRA| image:: _static/images/Logo_TRA1.png
        :width: 10 %
        :target: https://www.uni-bonn.de/research/research-profile/mathematics-modelling-and-simulation-of-complex-systems-1

      .. |DIW| image:: _static/images/Logo_DIW_Berlin.svg
        :width: 20 %
        :target: https://github.com/OpenSourceEconomics/respy/blob/main/docs/_static/funding/Becker_Sebastian_Arbeitsprogramm.pdf

      .. |space| raw:: html

           <embed>
           &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
           </embed>



   .. tab:: Citation

      **respy** was completely rewritten in the second release and evolved into a general
      framework for the estimation of Eckstein-Keane-Wolpin models. Please cite it with

      .. code-block::

          @Unpublished{Gabler2020,
            Title  = {respy - A Framework for the Simulation and Estimation of
                      Eckstein-Keane-Wolpin Models.},
            Author = {Janos Gabler and Tobias Raabe},
            Year   = {2020},
            Url    = {https://github.com/OpenSourceEconomics/respy},
          }          

      Before that, **respy** was developed by Philipp Eisenhauer and provided a package for
      the simulation and estimation of a prototypical finite-horizon discrete choice dynamic
      programming model. At the heart of this release is a Fortran implementation with Python
      bindings which uses MPI and OMP to scale up to HPC clusters. It is accompanied by a pure
      Python implementation as teaching material. If you use **respy** up to version 1.2.1,
      please cite it with

      .. code-block::

          @Software{Eisenhauer2019,
            Title  = {respy - A Package for the Simulation and Estimation of a prototypical
                      finite-horizon Discrete Choice Dynamic Programming Model.},
            Author = {Philipp Eisenhauer},
            Year   = {2019},
            DOI    = {10.5281/zenodo.3011343},
            Url    = {https://doi.org/10.5281/zenodo.3011343}
          }          

      We appreciate citations for **respy** because it helps us to find out how people have
      been using the package and it motivates further work.
