.. _installation:

Installation
------------

You'll need to install our package and git-annex, following the
instructions we provide next.

Install DataLad
~~~~~~~~~~~~~~~

In order to store and deliver the datasets, we recommend the use of DataLad,
"a decentralized system for integrated discovery, management, and publication of digital objects of science",
which we already use for several corpora.

DataLad is a python package, but it also requires git-annex to be installed.

For most MacOS and Linux users, these can be installed with:

.. code:: bash

    pip3 install datalad
    apt-get install git-annex || brew install git-annex


Detailed instructions for Windows, Linux and MacOS users are given in 
`DataLad's handbook <http://handbook.datalad.org/en/latest/intro/installation.html>`_,
including instructions to install DataLad via conda.

.. note::

    After installing, please make sure your version of git-annex
    is recent (we recommend 8.2 and later), with ``git-annex version``.
    
    
At this point, you may have received a message like the following:

> It is highly recommended to configure Git before using DataLad. Set both 'user.name' and 'user.email' configuration variables.

You can squash this message by providing these credentials (and if you already have an account on GitHub or GitLab, you can take your name and email from your GitHub or GitLab accounts; otherwise, just provide your name and email):

```
git config --global user.name "FIRST_NAME LAST_NAME"
git config --global user.email "MY_NAME@example.com"
```

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

Install ffmpeg
~~~~~~~~~~~~~~

Operations on the audio will require ffmpeg.
It is usually installed on most systems. You can check by typing ``ffmpeg -version``.
If not, it can be installed via ``apt install ffmpeg`` (Linux) or
``brew install ffmpeg`` (Mac), or ``conda install ffmpeg`` for conda users.

Check the setup
~~~~~~~~~~~~~~~

You can now make sure the packages have been successfully installed:

.. clidoc::

    datalad --version

.. clidoc::

   child-project --help


.. note::

    We recommend that you regularly keep DataLad and our package up to date. 
    To force-upgrade this package, do ``pip3 uninstall ChildProject``
    and then ``pip3 install git+https://github.com/LAAC-LSCP/ChildProject.git``.
    
    DataLad can be upgraded with ``pip3 install datalad --upgrade``
    (see DataLad's documentation for more details).
