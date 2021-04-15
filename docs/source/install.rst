.. _installation:

Installation
------------

You'll need to install our package and git-annex, following the
instructions we provide next.

Install our package via pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package can be installed using pip :

::

    pip3 install git+https://github.com/LAAC-LSCP/ChildProject.git

.. warning::
    
    Important: the package requires Python 3+. If the above command 
    does not work, try `pip` instead of `pip3`, but make sure that `pip`
    points to a python3 installation. You can do so by typing
    ``pip --version``. If you have accidentally installed the package on  
    the wrong environment, remove it with ``pip uninstall datalad``.

If you are having permissions issues - or any other issue -, you can try
any of the following :

-  Use the ``--user`` flag:

::

    pip3 install git+https://github.com/LAAC-LSCP/ChildProject.git --user

-  Create and activate a virtual python environment beforehand. You will
   have to activate it everytime you need to use the package or datalad.

::

    python3 -m venv ~/ChildProjectVenv
    source ~/ChildProjectVenv/bin/activate
    pip3 install git+https://github.com/LAAC-LSCP/ChildProject.git

Install git-annex
~~~~~~~~~~~~~~~~~

The package also requires git-annex in order to retrieve the datasets.
It can be installed via ``apt install git-annex`` (Linux) or
``brew install git-annex`` (Mac). It is encouraged to use a recent version of
git-annex (8.2+).

Install ffmpeg
~~~~~~~~~~~~~~

Operations on the audio will require ffmpeg.
It is usually installed on most systems. You can check by typing ``ffmpeg -version``.
If not, it can be installed via ``apt install ffmpeg`` (Linux) or
``brew install ffmpeg`` (Mac).

Check the setup
~~~~~~~~~~~~~~~

You can check the setup by issuing the following commands :

.. clidoc::

   datalad --version

.. clidoc::

   child-project --help


.. note::

    We recommend that you regularly keep DataLad and our package up to date. 
    This can be achieved with the commands ``pip3 install datalad --upgrade``
    and ``pip3 install git+https://github.com/LAAC-LSCP/ChildProject.git --upgrade``.