Contributing
============

Great, you are interesting in contributing to the package. Please announce yourself on our `mailing list <https://groups.google.com/forum/#!forum/respy/join>`_  so we can find you something to work on.

To get acquainted with the code base, check out our `issue tracker <https://gitlab.com/restudToolbox/package/issues>`_ for some immediate and clearly defined tasks. For more involved contributions, please see our roadmap below. Feel free to set up your development infrastructure using our `Amazon Machine Image <https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#LaunchInstanceWizard:ami=ami-6457c773>`_ or `Chef cookbook <https://github.com/restudToolbox/chef-respy>`_.

Roadmap
--------

We aim for improvements in three broad domains: Economics, Software Engineering, and Numerical Methods.

Economics
^^^^^^^^^

* add unobserved heterogeneity to the model as in Keane (1996)
* integrate codes from Eisenhauer (2016b) for decision making under ambiguity

Software Engineering
^^^^^^^^^^^^^^^^^^^^

* explore *Autotools* as a new build system
* research the *hypothesis* package to replace the hand-crafted property-based testing routines

Numerical Methods
^^^^^^^^^^^^^^^^^

* link the package to optimization toolkits such as *TAO* or *HOPSPACK*
* implement additional integration strategies following Judd (2011)
