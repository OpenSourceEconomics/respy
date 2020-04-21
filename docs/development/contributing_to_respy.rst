Contributing to respy
=====================

Contributions are always welcome. Everything ranging from small extensions of the
documentation to implementing new features of the model is appreciated. Of course, the
bigger the change the more it is recommended to reach out to us in advance for a
discussion. You can post an issue or use the email contact details under
:ref:`about_us`.


Where to start?
---------------

In general, you can take a look at our `issue tracker <https://github.com/
OpenSourceEconomics/respy/issues>`_ or at our :ref:`roadmap` to find opportunities to
contribute to the project.

If you are new to **respy**, you might want to check out issues labeled with
`good-first-issue <https://github.com/OpenSourceEconomics/respy/issues?q=is%3Aissue+is
%3Aopen+label%3Agood-first-issue>`_.


Workflow
--------

1. Assuming you have settled on contributing a small fix to the project, fork the
   `repository <https://github.com/OpenSourceEconomics/respy/>`_. This will create a
   copy of the repository where you have write access. Your fix will be implemented in
   your copy. After that, you will start a pull request (PR) which means a proposal to
   merge your changes into the project.

2. Clone the repository to your disk. Set up the environment of the project with conda
   and the ``environment.yml``. Implement the fix.

3. We validate contributions in three ways. First, we have a test suite to check the
   implementation of **respy**. Second, we correct for stylistic errors in code and
   documentation using linters. Third, we test whether the documentation builds
   successfully.

   You can run all checks with ``tox`` by running

   .. code-block:: bash

       $ tox

   This will run the complete test suite. To run only a subset of the suite you can use
   the environments, ``pytest``, ``linting`` and ``sphinx``, with the ``-e`` flag of
   tox.

   Correct any errors displayed in the terminal.

   To correct stylistic errors, you can also install the linters as a pre-commit with

   .. code-block:: bash

       $ pre-commit install

   Then, all the linters are executed before each commit and the commit is aborted if
   one of the check fails. You can also manually run the linters with

   .. code-block:: bash

       $ pre-commit run -a

4. If the tests pass, push your changes to your repository. Go to the Github page of
   your fork. A banner will be displayed asking you whether you would like to create a
   PR. Follow the link and the instructions of the PR template. Fill out the PR form to
   inform everyone else on what you are trying to accomplish and how you did it.

   The PR also starts a complete run of the test suite on a continuous integration
   server. The status of the tests is shown in the PR. Reiterate on your changes until
   the tests pass on the remote machine.

5. Ask one of the main contributors to review your changes. Include their remarks in
   your changes.

6. If the changes should be merged, add yourself to the list of contributors and
   depending on the size of the changes, add a note to ``CHANGES.rst``. The final PR
   will be merged by one of the main contributors.


Contributing to the documentation
---------------------------------

The documentation is written in **reStructuredText** or as a Jupyter notebook and
rendered with `Sphinx <https://www.sphinx-doc.org>`_ whose documentation also provides
an `introduction to reST
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_.

The correct code formatting in ``.rst`` files is ensured by ``blacken-docs``. To
correctly format the code inside notebooks, use the `jupyterlab-code-formatter
<jupyterlab-code-formatter.readthedocs.io>`_.


Types of documents
~~~~~~~~~~~~~~~~~~

Tutorials, explanations, how-to guides, and reference guides are the cornerstones of the
documentation and, thus, there are stronger requirements in place for a contribution.
The structure is inspired by Daniele Procida who shared his approach in `text and video
<https://documentation.divio.com/>`_. The main take-aways for each part are:

1. Tutorials are learning-oriented and should guide the user through a series of clearly
   defined steps with little fuss. Every step needs to produce a result to build the
   user's self-confidence in using the software. Offer only the minimum necessary
   explanation. Provide some links to related explanations or how-to guides, but do
   overdo it and create confusion.

2. Explanations are understanding-oriented. They provide context, discuss alternatives,
   but most importantly, nothing is done nor is the code discussed directly.

3. How-to guides are problem-oriented and offer a series of steps to achieve a specific
   result. They do not explain, they leave things out.

4. Reference guides are information-oriented and explain the implementation and should
   be read with the source code. They have preferably the same structure as the code.

The following figure shows how the sections relate to each other and that some have a
natural proximity. Though, it is important to keep the separation in place for a well
organized documentation.

.. image:: https://documentation.divio.com/_images/overview.png
   :width: 70%


Styleguide for the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- In general, follow the style applied in other documents.
- Use the following order of headings

   + ``===``
   + ``---``
   + ``~~~``
   + ``^^^``
   + ``"""``
- Between the end of a section and the following heading are two empty lines.


Contributing docstrings
-----------------------

Docstrings in **respy** are written in `NumPy Docstring Standard
<https://numpydoc.readthedocs.io/en/latest/format.html>`_.


Styleguide for docstrings
~~~~~~~~~~~~~~~~~~~~~~~~~

- The first line of a docstring starts is right after the three quotes.
- Keep a newline between the last text of a docstring and the closing quotes.
- Inline code is surrounded by single backticks, e.g., ```sum```.


Contributing to the code base
-----------------------------

The style guide for the code base is enforced by several linters and formatters which do
not leave much room for idiosyncrasies.

The user must focus on

- using code simple structures if possible
- meaningful variable names
