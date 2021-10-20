.. _installation:

Installation
------------

The following instructions will let you install two python packages:

 - **ChildProject**, the package that is documented here.
 - **DataLad**, a python software for the management and delivery of scientific data. Although ChildProject may work without it, a number of datasets of daylong recordings of children require it.

Linux users
~~~~~~~~~~~

.. code:: bash

    # download the conda environment
    wget https://raw.githubusercontent.com/LAAC-LSCP/ChildProject/master/env_linux.yml -O env.yml

    # create the conda environment
    conda env create -f env.yml

    # activate the environment (this should be done systematically to use our package)
    conda activate childproject


MacOS users
~~~~~~~~~~~

.. code:: bash

    # download the conda environment
    curl https://raw.githubusercontent.com/LAAC-LSCP/ChildProject/master/env_macos.yml -o env.yml

    # create the conda environment
    conda env create -f env.yml

    # activate the environment (this should be done systematically to use our package)
    conda activate childproject

    # install git-annex from brew
    brew install git-annex


Check the setup
~~~~~~~~~~~~~~~

You can now make sure the packages have been successfully installed:

.. clidoc::

   child-project --help

.. clidoc::

    # optional software, for getting and sharing data
    datalad --version

.. note::

    We recommend that you regularly keep DataLad and our package up to date. 
    To force-upgrade this package, do ``pip uninstall ChildProject``
    and then ``pip install ChildProject --upgrade``.

    You may also want to install the development version from GitHub in order
    to receive more recent updates before their release:

    .. code:: bash
    
        pip install git+https://github.com/LAAC-LSCP/ChildProject.git --force-reinstall

    Since some updates may break compatibility with previous versions,
    we advise you to read the `Change Log <https://github.com/LAAC-LSCP/ChildProject/blob/master/CHANGELOG.md>`_
    before upgrading.
    
    DataLad can also be upgraded with ``pip install datalad --upgrade``
    (see DataLad's documentation for more details).

Troubleshooting
~~~~~~~~~~~~~~~

ChildProject
============

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

DataLad
=======


In case DataLad does not work, please refer to its detailed installation instructions for Windows, Linux and MacOS users are given in 
`DataLad's handbook <http://handbook.datalad.org/en/latest/intro/installation.html>`_,
including instructions to install it via conda.
    
If DataLad prints the following message:

> It is highly recommended to configure Git before using DataLad. Set both 'user.name' and 'user.email' configuration variables.

You can squash this message by providing these credentials (and if you already have an account on GitHub or GitLab, you can take your name and email from your GitHub or GitLab accounts; otherwise, just provide your name and email):

```
git config --global user.name "FIRST_NAME LAST_NAME"
git config --global user.email "MY_NAME@example.com"
```

.. warning::

    Mac OS <= 10.13 (High Seria) does not support DataLad to our knowledge.
    You will need to upgrade your OS to a later version.