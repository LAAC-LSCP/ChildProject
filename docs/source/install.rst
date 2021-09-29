.. _installation:

Installation
------------

Installing our package
~~~~~~~~~~~~~~~~~~~~~~

The package can be installed using pip :

::

    pip3 install ChildProject

.. warning::
    
    Important: the package requires Python 3+. If the above command 
    does not work, try `pip` instead of `pip3`, but make sure that `pip`
    points to a python3 installation. You can do so by typing
    ``pip --version``. If you have accidentally installed the package on  
    the wrong environment, remove it with ``pip uninstall datalad``.

If you are having permissions issues - or any other issue -, you can
install the package in a virtual python environment beforehand. You will
have to activate it everytime you need to use the package or datalad.

::

    python3 -m venv ~/ChildProjectVenv
    source ~/ChildProjectVenv/bin/activate
    pip3 install ChildProject

Install ffmpeg
==============

Some of the tools for audio processing (e.g. conversion to another format)
will require ffmpeg.
It is usually installed on most systems. You can check by typing ``ffmpeg -version``.

**If** it is not installed, it can be installed via ``apt install ffmpeg`` (Linux) or
``brew install ffmpeg`` (Mac), or ``conda install ffmpeg`` for conda users.

Installing DataLad
~~~~~~~~~~~~~~~~~~

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

.. warning::

    Mac OS <= 10.13 (High Seria) does not support DataLad to our knowledge.
    You will need to upgrade your OS to a later version.

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
    and then ``pip3 install ChildProject --upgrade``.

    You may also want to install the development version from GitHub in order
    to receive more recent updates before their release:

    .. code:: bash
    
        pip3 install git+https://github.com/LAAC-LSCP/ChildProject.git --force-reinstall

    Since some updates may break compatibility with previous versions,
    we advise you to read the `Change Log <https://github.com/LAAC-LSCP/ChildProject/blob/master/CHANGELOG.md>`_
    before upgrading.
    
    DataLad can also be upgraded with ``pip3 install datalad --upgrade``
    (see DataLad's documentation for more details).

Troubleshooting
~~~~~~~~~~~~~~~

If you are having trouble installing ChildProject, please look
for similar issues on our GithHub (in `Issues <https://github.com/LAAC-LSCP/ChildProject/issues>`__ or `Discussions <https://github.com/LAAC-LSCP/ChildProject/discussions>`__).

If this issue is related to a dependency of the package, we recommend that you ask
the developers of the depdendency directly as you may get more accurate advice.

If this issue is related to DataLad, please create an issue on `DataLad's GitHub <https://github.com/datalad/datalad/issues>`__.

.. warning::

    ChildProject is only officially supported on Linux and Mac for python >= 3.7.
    We perform automated, continuous testing on these environments to look
    for potential issues at any step of the development.

    We expect the package to work on Windows, although we do not perform
    automated tests on this OS at the moment.