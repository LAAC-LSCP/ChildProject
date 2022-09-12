.. _installation:

Installation
------------

The following instructions will let you install two python packages:

 - **ChildProject**, the package that is documented here.
 - **DataLad**, a python software for the management and delivery of scientific data. Although ChildProject may work without it, a number of datasets of daylong recordings of children require it.

.. note::

    The default installation procedure requires anaconda. If you are not sure you have conda installed, please do ``conda --version``.
    If you don't, please refer to the instructions `here <https://docs.anaconda.com/anaconda/install/index.html>`_.

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

.. note::

    The ChildProject package alone can also be installed directly through pip, without using conda.
    However, this means git-annex, ffmpeg, and other dependencies that are not installable
    through pip will have to be installed by hand.

    The following command will install the python package alone via pip and pypi:

    .. code:: bash

        pip install ChildProject

Windows users
~~~~~~~~~~~~~

.. warning::

    ChildProject is only officially supported on Linux and Mac for python >= 3.7.
    We perform automated, continuous testing on these environments to look
    for potential issues at any step of the development.

    We expect the package to work on Windows, although we do not perform
    automated tests on this OS at the moment.

    If you are on a Windows system, consider using a `Windows subsystem for linux (WSL) <https://docs.microsoft.com/en-us/windows/wsl/install>`_,
    inside of which you can use all the Linux instructions while accessing your windows filesystem.

If you which to continue using directly Windows, you will first need to `install git-annex <https://git-annex.branchable.com/install/Windows/>`_. 
Follow the instructions by first `installing Git for Windows <http://git-scm.com/downloads>`_ and once Git is installed, downloading and running the `git-annex installer <https://downloads.kitenet.net/git-annex/windows/current/>`_.

Now, install Miniconda to manage your environments by following `this guide <https://conda.io/projects/conda/en/latest/user-guide/install/windows.html>`_.

Then, Open a GIT CMD prompt (search for "GIT CMD" in the start Menu)
and run the following commands to create your environment:

.. code:: bash

    # download the conda environment creation info
    curl https://raw.githubusercontent.com/LAAC-LSCP/ChildProject/master/env_macos.yml -o env.yml

    # create the conda environment, keep the default parameters, this may take a long time
    conda env create -f env.yml

    # activate the environment (this should be done systematically to use our package)
    conda activate childproject

    #Your command prompt should be prefixed by '(childproject)' at this point


Check the setup
~~~~~~~~~~~~~~~

You can now make sure the packages have been successfully installed:
Each --version command should output the version of the package

.. clidoc::

   child-project --help

.. clidoc::

    child-project --version

.. clidoc::

    git annex version

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

If you are having trouble installing ChildProject, please look
for similar issues on our GithHub (in `Issues <https://github.com/LAAC-LSCP/ChildProject/issues>`__ or `Discussions <https://github.com/LAAC-LSCP/ChildProject/discussions>`__).

If this issue is related to a dependency of the package, we recommend that you ask
the developers of the depdendency directly as you may get more accurate advice.

If this issue is related to DataLad, please create an issue on `DataLad's GitHub <https://github.com/datalad/datalad/issues>`__.


Frequently Asked Questions
~~~~~~~~~~~~~~~~~~~~~~~~~~

    *I don't have anaconda and I can't install it. What should I do?*

You should try to install the package inside of a python environment instead, e.g.:

.. code:: bash

    python3 -m venv ~/ChildProjectVenv
    source ~/ChildProjectVenv/bin/activate
    pip3 install ChildProject

You will still need git-annex in order to use DataLad.
It can be installed with brew for Mac users (`brew install git-annex`)
or through apt for Linux users (`apt install git-annex`).
Most likely, you will lack permissions to do so if you failed to install anaconda.
In this case, pleaser refer to your system administrator.

    *``brew install git-annex`` does not work!*

Please try ``brew install --build-from-source git-annex``. 

If this does not work better for you, make sure that your version of Mac OS is 10.14 or later.
We are aware of issues with Mac OS 10.13 (High Sierra) and earlier.

If your issues persistent, please report it to [DataLad](https://github.com/datalad/datalad).