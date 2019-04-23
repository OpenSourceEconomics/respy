Releases
========

What is the new version number?
-------------------------------

The version number depends on the severity of the changes and adheres to `semantic
versioning <https://semver.org/>`_. The format is x.y.z.

You are also allowed to append ``-rc.1`` after the last digit to indicate the first or
higher release candidates. Thus, you can test deployment on PyPI and release preliminary
versions.


How to release a new version?
-----------------------------

1. At first, we can draft a release on Github. Go to
   https://github.com/OpenSourceEconomics/respy/releases and click on "Draft a new
   release". Fill in the new version number as a tag and title. You can write a summary
   for the release, but also do it later. Important: Only save the draft. Do not publish
   yet.

2. Second, we need to create a final PR to prepare everything for the new version. The
   name of the PR and the commit message will be "Release vx.y.z". We need to

   - update all references of the old version number (``setup.py``,
     ``respy/__init__.py``, ``docs/conf.py``).
   - update information in ``CHANGES.rst`` to have summary of the changes which
     can also be posted in the Github repository under the tag.

   Merge the PR into master.

3. After that, revisit the draft of the release. Make sure everything is fine. Now, you
   click on "Publish release" which creates a version tag on the latest commit of the
   specified branch. The tag will trigger a build on Travis-CI which will publish the
   release on PypI.

4. Make sure that the new release was indeed published by checking `PyPI
   <https://pypi.org/project/respy/>`_.

4. Spread the word!


Notes
-----

- Travis-CI only builds tags if "Build pushed branches" is active.
- If you publish a release on PyPI, the same version number cannot be reused even if you
  delete the release. This is a safety measure. If you are not sure whether the release
  will work, create a release candidate instead and publish the real version later.
