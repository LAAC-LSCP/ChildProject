Installation
------------

You'll need to install our package and git-annex, following the
instructions we provide next.

Install our package via pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package can be installed using pip :

::

    pip install git+https://github.com/LAAC-LSCP/ChildRecordsData.git

Important: the package requires Python 3+, so make sure ``pip`` points
to a python3 installation, otherwise try pip3. You can do so by typing
``pip --version``.

If you are having permissions issues - or any other issue -, you can try
any of the following :

-  Use the ``--user`` flag:

::

    pip install git+https://github.com/LAAC-LSCP/ChildRecordsData.git --user

-  Create and activate a virtual python environment beforehand. You will
   have to activate it everytime you need to use the package or datalad.

::

    python3 -m venv ~/ChildProjectVenv
    source ~/ChildProjectVenv/bin/activate
    pip install git+https://github.com/LAAC-LSCP/ChildRecordsData.git

Install git-annex
~~~~~~~~~~~~~~~~~

The package also requires git-annex in order to retrieve the datasets.
It can be installed via ``apt install git-annex`` or
``brew install git-annex``. It is encouraged to use a recent version of
git-annex (8.2+).

Check the setup
~~~~~~~~~~~~~~~

You can check the setup by issuing the following commands :

.. clidoc::

   datalad --version

.. clidoc::

   child-project --help