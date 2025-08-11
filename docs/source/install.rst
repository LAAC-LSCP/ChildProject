.. _installation:

Installation
============

ChildProject is a python package distributed through `pypi <https://pypi.org/project/ChildProject>`__. It is meant to
function in conjunction with multiple dependency packages, both python packages and system packages. This is why we
advise users to install all of those in a self-contained conda environment, this will ensure separation between your
system and the software used in projects using ChildProject. You can choose to install ChildProject on its own,
but be aware that some features may fail or be unavailable if some system packages used by ChildProject are missing.

Installation in a conda environment
-----------------------------------

The following instructions will let you install the following packages inside a self-contained conda environment:

 - **ChildProject** (and dependencies), the package that is documented here, follow the main installation instructions to get it
 - **DataLad** (and dependencies), a python software for the management and delivery of scientific data. Although ChildProject may work without it, a number of datasets of daylong recordings of children require it. We advise to work with it alongside ChildProject. Follow the instructions in the section "Datalad installation" to get it as well.

This installation procedure requires a package able to handle conda environments. We recommend using `micromamba <https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html>`__
but you can use any alternative. A very common alternative is anaconda (conda command). If you are not sure you have conda or
micromamba installed, check the :ref:`environment_manager_section`.

.. warning::

    All our instructions use ``micromamba`` commands, if you are using a difference environment manager, such as anaconda
    , replace all instructions with the proper command (``conda`` for example).

.. Automatic install
.. ~~~~~~~~~~~~~~~~~
..
.. Reminder, for running the automatic install, you need an :ref:`environment_manager_section` environment manager to be installed
..
.. .. tabs::
..
..    .. group-tab:: linux
..
..       To install automatically in Linux run in a terminal
..
..       .. code-block:: bash
..
..          "${SHELL}" <(curl -L https://raw.githubusercontent.com/LAAC-LSCP/ChildProject/master/install.sh)
..
..    .. group-tab:: MacOS
..
..       To install automatically in MacOS run in a terminal
..
..       .. code-block:: bash
..
..          "${SHELL}" <(curl -L https://raw.githubusercontent.com/LAAC-LSCP/ChildProject/master/install.sh)
..
..    .. group-tab:: Windows
..
..       To install automatically in Linux, MacOS run in a terminal
..
..       .. code-block:: powershell
..
..          Invoke-Expression ((Invoke-WebRequest -Uri https://raw.githubusercontent.com/LAAC-LSCP/ChildProject/master/install.ps1 -UseBasicParsing).Content)
..
..
.. Manual install
.. ~~~~~~~~~~~~~~

.. tabs::

   .. group-tab:: linux

      .. code-block:: bash

        # download the conda environment
        curl https://raw.githubusercontent.com/LAAC-LSCP/ChildProject/master/env_linux.yml -o env.yml

        # create the conda environment
        micromamba env create -f env.yml

        # activate the environment (this should be done systematically to use our package)
        micromamba activate childproject

   .. group-tab:: MacOS

      Use a terminal window

      .. code-block:: bash

         # download the conda environment
         curl https://raw.githubusercontent.com/LAAC-LSCP/ChildProject/master/env_macos.yml -o env.yml

         # create the conda environment
         micromamba env create -f env.yml

         # activate the environment (this should be done systematically to use our package)
         micromamba activate childproject


   .. group-tab:: Windows

      Use powershell if conda/micromamba is installed as a command, or use the anaconda/miniconda powershell prompt
      program if conda is installed as a program.

      .. code-block:: powershell

         # download the conda environment
         curl https://raw.githubusercontent.com/LAAC-LSCP/ChildProject/master/env_windows.yml -o env.yml

         # create the conda environment
         micromamba env create -f env.yml

         # activate the environment (this should be done systematically to use our package)
         micromamba activate childproject


Installation as system packages and separated python packages
-------------------------------------------------------------

Installing directly dependencies as system packages is not recommended as this won't clearly separate dependencies
between projects and requires you to have administrator privileges.

Instructions here will install the system dependencies directly into the operating system, the python requirements will
be installed separately (as installing python packages directly in the system is highly discouraged as
it could break system packages, still installing directly is possible by replacing `pipx` with `pip` and passing the
`--break-system-packages` flag.) In windows, those intructions guide you into installing the python packages in anaconda
prompt.


.. tabs::

   .. group-tab:: linux

      Make sure you have a recent version of python installed or install it (https://www.python.org/)

      Installing packages on Debian

      .. code-block:: bash

        sudo apt-get install sox ffmpeg git git-annex pipx

      Installing packages on Fedora/RHEL/CentOS/Rocky

      .. code-block:: bash

        # if the EPEL is not added on your system, you may need to add it with
        # sudo dnf install epel-release -y
        # sudo crb enable
        # sudo dnf install rpmfusion-free-release -y

        sudo dnf install sox ffmpeg git git-annex pipx

      Installing packages on Arch

      .. code-block:: bash

        sudo pacman -S sox ffmpeg git git-annex pipx

      Then python packages can be installed with pipx directly (or alternatives like uv)

      .. code-block:: bash

        pipx install datalad
        pipx install ChildProject

   .. group-tab:: MacOS

      Make sure you have a recent version of python installed or install it (https://www.python.org/)

      .. code-block:: bash

         # install via brew, git should be installed by default
         brew install sox ffmpeg git-annex pipx

         pipx install datalad
         pipx install ChildProject

   .. group-tab:: Windows

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


Check the setup
---------------

You can now make sure the packages have been successfully installed:
each --version command should output the version of the package.
If you installed in an environment, check it is activated or activate it

.. clidoc::

   child-project --version
   datalad --version

.. note::

    We recommend that you regularly keep DataLad and our package up to date. 
    To upgrade this package, do ``pip install ChildProject --upgrade`` in virtual environments, or
    ``pipx upgrade ChildProject`` if installed directly.

    You may also want to install the development version from GitHub in order
    to receive more recent updates before their release:

    .. code:: bash
    
        pip install git+https://github.com/LAAC-LSCP/ChildProject.git --force-reinstall

    Since some updates may break compatibility with previous versions,
    we advise you to read the `Change Log <https://github.com/LAAC-LSCP/ChildProject/blob/master/CHANGELOG.md>`__
    before upgrading.
    
    DataLad can also be upgraded with ``pip install datalad --upgrade``
    (see DataLad's documentation for more details).

.. _environment_manager_section:

Check and or install Anaconda / Micromamba
------------------------------------------

To check if conda or micromamba is installed, run in a terminal or powershell the following commands :
``conda --version`` and ``micromamba --version``. If either command prints a version number, they are installed and
you should be able to use them.

On windows, you may have a 'prompt' version of anaconda install, search for 'powershell anaconda prompt' in your
programs, launch it to use a terminal where cona is available.

If none is installed, refer to the installation instructions:

Micromamba
~~~~~~~~~~

Micromamba is a standalone executable that can be installed directly in a terminal using one instruction. On Windows,
 use preferably the Powershell option. Follow the instructions `here <https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#automatic-install>`__.
 Installing micromamba may not require admin privileges.

Miniconda
~~~~~~~~~

Miniconda is installed as a program, which means you need admin rights. It is then accessible in the terminal on
Linux/MacOS and by launching a program on Windows (miniconda prompt). Follow instructions `here <https://docs.anaconda.com/anaconda/install/index.html>`__
to install.


Troubleshooting
---------------

If you are having trouble installing ChildProject, please look
for similar issues on our GithHub (in `Issues <https://github.com/LAAC-LSCP/ChildProject/issues>`__ or `Discussions <https://github.com/LAAC-LSCP/ChildProject/discussions>`__).

If this issue is related to a dependency of the package, we recommend that you ask
the developers of the dependency directly as you may get more accurate advice.

If this issue is related to DataLad, please create an issue on `DataLad's GitHub <https://github.com/datalad/datalad/issues>`__.


Frequently Asked Questions
--------------------------

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

If your issues persistent, please report it to `DataLad <https://github.com/datalad/datalad>`__.
