Docker
======

We use Docker to create a reliable cross-platform environment in which our exhaustive
test battery can run.

Installation
------------

Download the appropriate Docker engine `here
<https://hub.docker.com/search/?type=edition&offering=community>`_.

.. warning:: You cannot install Docker on Windows 10 Home and on older MacOS. Choose the
             `Docker Toolbox <https://docs.docker.com/toolbox/overview/>`_ instead.



Build the image
---------------

In order to build the image with the current respy version in the repository, cd to the
respy folder and type

.. code-block:: bash

    $ docker build -t respy .

.. warning:: On Linux systems you will need sudo permissions for `docker build`

To be clear, the image is built with the respy version in the directory using ``pip
install .``. Unfortunately, this means that you have to rebuild the image every time you
change something in the code. Since Docker creates layers during builds which are
essentially snapshots after each statement in the ``Dockerfile``, this is not as costly
as it sounds.


Enter the container interactively
---------------------------------

Type

.. code-block:: bash

    $ docker run -it respy


.. warning:: On Linux systems you will need sudo permissions for `docker run`

to enter the container and open a bash terminal. Then, switch to the correct environment
with

.. code-block:: bash

    $ conda activate respy.

After entering the container you are in the home directory of a Linux system that
contains the standard files and the `respy` folder.

If you want to exit the container, hit ``Ctrl + d`` or type

.. code-block:: bash

    $ exit.

Reclaiming space on disk
------------------------

Docker occupies a lot of space on your disk due to saving snapshots of the container,
container itself, images, etc.. To reclaim the space, `prune
<https://docs.docker.com/config/pruning/>`_ unused docker objects. Make sure that you do
not accidentally delete valuable information.

.. code-block:: bash

    $ docker container prune  # Delete unused containers.
    $ docker system prune     # Delete unused docker objects.
