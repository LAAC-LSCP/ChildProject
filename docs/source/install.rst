.. _installation:

Installation
------------

The following instructions will let you install two python packages:

 - **ChildProject**, the package that is documented here.
 - **DataLad**, a python software for the management and delivery of scientific data. Although ChildProject may work without it, a number of datasets of daylong recordings of children require it.

.. note::

    The default installation procedure requires anaconda. If you are not sure you have conda installed, please do `conda --version`.
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


Frequently Asked Questions
~~~~~~~~~~~~~~~~~~~~~~~~~~

    I don't have anaconda and I can't install it. What should I do?

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