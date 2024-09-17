.. _installation:

Installation
------------

The following instructions will let you install two python packages:

 - **ChildProject**, the package that is documented here, follow the main installation instructions to get it
 - **DataLad**, a python software for the management and delivery of scientific data. Although ChildProject may work without it, a number of datasets of daylong recordings of children require it. We advise to work with it alongside ChildProject. Follow the instructions in the section "Datalad installation" to get it as well.

.. note::

    The default installation procedure requires anaconda. If you are not sure you have conda installed, please do ``conda --version``.
    If you don't, please refer to the instructions `here <https://docs.anaconda.com/anaconda/install/index.html>`__.

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

    If you are on a Windows system, consider using a `Windows subsystem for linux (WSL) <https://docs.microsoft.com/en-us/windows/wsl/install>`__,
    inside of which you can use all the Linux instructions while accessing your windows filesystem.

If you which to continue using directly Windows, you must do the following:
 1. Download and run the Git installer found in `this page <https://git-scm.com/download/win>`__, use the first link in the page. When running the installer, we advise you keep all the default choices.
 
 .. figure:: images/git-install.png
    :height: 300
    :alt: git installer Wizard

    git installer

 2. Download and run the git-annex installer found `here <https://downloads.kitenet.net/git-annex/windows/current/>`__, download the file 'git-annex-installer.exe' and then launch it, keep everything as default.
 
 .. figure:: images/git-annex-install.png
    :height: 300
    :alt: git annex installer Wizard

    git annex installer

 3. Download and run the `Miniconda installer <https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe>`__, launch the installer and keep all the default options.
 
 .. figure:: images/miniconda-install.png
    :height: 300
    :alt: miniconda installer Wizard

    miniconda installer

 4. Open an Anaconda prompt, after all the installations, you should now have a program called "Anaconda Prompt" in your start Menu, if you can't find it, use the search field. You will use this program whenever you use ChilProject so it is probably best to pin it to the start menu or create a shortcut on your desktop. Launch it, you should be presented with a terminal window, allowing you to enter and launch commands
 
 .. figure:: images/anaconda-prompt.png
    :alt: Anaconda prompt cmd

    Anaconda prompt

 5. Use the following command to download the environment description

 .. code:: bash

     # download the conda environment creation info
     curl https://raw.githubusercontent.com/LAAC-LSCP/ChildProject/master/env_macos.yml -o env.yml

 
 .. figure:: images/download-yml.png
    :alt: download environment description file

    download the conda environment description file

 6. Run this line to create the conda environment, keep the default parameters, this can take several minutes 

 .. code:: bash

     # create the conda environment, keep the default parameters, this may take a long time
     conda env create -f env.yml
 
 .. figure:: images/env-install.png
    :alt: creation of conda environment

    creation of the environment

 7. Activate the childproject environment in your Anaconda Prompt. This must be done everytime you use childproject

 .. code:: bash

     # activate the environment (this should be done systematically to use the package)
     conda activate childproject
 
 .. figure:: images/env-activate.png
    :alt: activate the childproject environment

    activate the newly created environment, to do every time we launch a new anaconda prompt

Congratulations, You are now able to use all the childproject features inside your Anaconda Prompt.

Datalad installation
~~~~~~~~~~~~~~~~~~~~

This section is optional, it will help you to install Datalad once you have completed the previous step of installing ChildProject.

If you followed the previous instructions correctly, you should have created and activated a conda environment with ChildProject installed. When you are in this environment, run the following:

.. code:: bash

     # install datalad in your environment
     pip install datalad


Check the setup
~~~~~~~~~~~~~~~

You can now make sure the packages have been successfully installed:
Each --version command should output the version of the package

.. clidoc::

   child-project --help

.. clidoc::

    # run this ONLY IF you have installed the optional software datalad, for getting and sharing data
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
