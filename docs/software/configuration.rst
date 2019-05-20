Configuration
=============

``respy`` adheres to the ` Twelve-Factor App <https://12factor.net/>`_ methodology and
exposes its configuration via environment variables which have to be set prior to
importing ``respy`` via

.. code-block:: python

    import os


    os.environ["<key>"] = "<value>"


The following options are available:

* ``RESPY_DEBUG`` is ``False`` by default and controls the output of additional
  information for debugging purposes.
* ``RESPY_KW_SQUARED_EXPERIENCES`` is ``False`` by default which means that wages are
  computed with quadratic experiences divided by 100. For the original models by Keane
  and Wolpin, the division was omitted.
