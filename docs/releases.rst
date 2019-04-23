Releases
========

What is the new version number?
-------------------------------

The version number depends on the severity of the changes and adheres to `semantic
versioning <https://semver.org/>`_.

You are also allowed to append ``rc1`` after the last digit to indicate the first or
higher release candidates. By that, you can test the release and deployment on PyPI and
release preliminary versions.


How to release a new version?
-----------------------------

1. At first, we need to create a new PR to prepare everything for the new version. We
   need to

   - update all mentions of the old version number
   - update information in ``CHANGES.rst`` to have summary of the changes which
     can also be posted in the Github repository under the tag.

2. After the PR is merged into master, go to the master branch in your local repository
   and pull the latest changes. Make sure that the current tip of the branch corresponds
   to the state where you want to set the new version. Then, type

   .. code-block:: bash

       $ git tag -m "x.x.x"

   to create a tag with the version number. After that, you need to push the tag to the
   remote repository which triggers a Travis-CI build and deployment to PyPI.

   .. code-block:: bash

       $ git push --tags

3. Make sure that the new release was indeed published by checking `PyPI
   <https://pypi.org/project/respy/>`_. Also, copy the information from the new release
   in ``CHANGES.rst`` and post it under the `new release
   <https://github.com/OpenSourceEconomics/respy/releases/>`_.

4. Spread the word!
