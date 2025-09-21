A project from scratch
======================

.. toctree::
   :maxdepth: 4

Here we give an example of creating a dataset from scratch. You may want to revisit
this page with more experience, or if you want a whirlwind tour, follow the steps
with or without a full comprehension.

We will try to show you a little bit of everything
1. How to create a dataset
2. How to load data
3. How to run our models on these files and move outputs into the right folders

We assume familiarity with the bash shell. I think what matters most is that you know
the following commands: ``mv``, ``ls``, ``cd``, ``mkdir``.

We assume you have ChildProject and git-annex installed, alongside miniconda, as per the installation instructions.
You may want to install DataLad if you follow the steps of getting data. For this make sure you're in the conda environment
and run ``pip install datalad``. If you already have data, however, you won't need to do this.

Getting data (in case you don't have any)
-----------------------------------------
Let's get some data. Ignore this if you already have data.
Let us start with the Vandam-Daylong dataset. This is a miniature example dataset.

This tutorial intends to show you how to start from scratch. But Vandam-Daylong already comes packed as a complete dataset.
We only want the recordings. Unfortunately, but for good reasons, Vandam-Daylong requires DataLad to fetch the recordings.

Without going into the details of DataLad or the GIN platform, go into a clean folder somewhere and type:

.. code-block:: bash

    datalad clone https://gin.g-node.org/LAAC-LSCP/vandam-data

Now cd into your vandam-data folder `cd vandam-data`. Run `datalad get recordings/raw/**` to get the raw recordings. There is only one.

.. note::

    While we're jumping in and across datasets, I will always make sure that the present working directory (pwd) is no deeper than the root of the
    dataset. Otherwise there's lots of jumping around folders, and it's easy to get lost.

The raw recordings are actually stored on the GIN servers or elsewhere. This would not typically be the case for your own datasets. In
fact, for this reason, the recording file itself is actually a symbolic link, a sort of pointer to a file somewhere else on your computer,
inside an "annex". To mimick a more realistic setup, let's rip the recording out of the annex ``git annex unlock recordings/raw/BN32_010007.mp3``.
If this is all confusing--and it surely once was for me--just run the commands and trust the process.

At this point we're still in the vandam-data folder. Let's step out of it, into the parent folder with ``cd ..``. The next step will be to
actually make our dataset.

Create the dataset
------------------

Create the dataset with ``mkdir dataset-from-scratch``. At this point it's just an empty folder. Step into it with ``cd dataset-from-scratch``.

Now let's set up the boilerplate using ChildProject. Run ``child-project init .``. If you run ``tree``, you will see that a few folders and files were created.
In particular, we have our annotations, extra, metadata, recordings and scripts folder. There is a children.csv and recordings.csv file, both empty
except for their column headers. Running ``child-project validate .`` verifies that our dataset is in a clean, albeit pretty empty, state.

.. note::

    If you're using Datalad as we are for version control and (large) file management, it's recommended to run ``datalad create`` before you run any other
    commands. This turns your dataset into a datalad repository (sort of a super-powered git repository).

Adding raw recordings
~~~~~~~~~~~~~~~~~~~~~

The next step is to put all your raw recordings under the recordings/raw folder. If you've followed the steps for the Vandam-Daylong data to a tee, you'll be albe
to run 

.. code-block:: bash

    mv ../vandam-data/recordings/raw/BN32_010007.mp3 ./recordings/raw/BN32_010007.mp3

to move the vandam-data raw recording into the raw recordings folder. If you have other recordings use those instead. Feel free to drag and drop instead of using ``mv``.

Next Childproject needs to be made aware of these recordings. This digital awareness is all achieved in the metadata. In ``recordings.csv``,
we will add. Currently, ``recordings.csv`` has the following columns: experiment,child_id,date_iso,start_time,recording_device_type,recording_filename
Let us call this experiment ``dataset_from_scratch``, use child_id ``CHI_01``, date_iso ``2025-09-20``, start_time ``08:00:00``, recording_device_type ``unknown``,
and recording_filename ``BN32_010007.mp3``.

.. note::
    Well, that is assuming you're using the Vandam-Data data... Otherwise you will need to add many more rows with the corresponding filenames.

To add a single row to this file run

.. code-block:: bash

    echo dataset_from_scratch,CHI_01,2025-09-20,08:00:00,unknown,BN32_010007.mp3 >> metadata/recordings.csv

Or use your favorite text editor.

To inspect the contents of the file run ``cat metadata/recordings.csv`` and check that all is correct. Now the recording has a reference to a child
with id ``CHI_01`` and experiment ``dataset_from_scratch``, which to ChildProject makes no sense, as no such child is registered in the children metadata.
``children.csv`` contains the fields experiment,child_id,child_dob. So run e.g.,

.. code-block:: bash

    echo dataset_from_scratch,CHI_01,2020-10-08 >> metadata/children.csv

Now run ``child-project validate .`` to check if everything is tied up correctly. As an odd side effect, running this command creates the
``metadata/annotations.csv`` file as well, which we will need moving forward.

Converting recordings
~~~~~~~~~~~~~~~~~~~~~
The models we run, such as VTC, or ALICE, are trained on audio at 16,000Hz. As a general rule we convert all audio even if
the sampling rate is already at 16,000Hz.

First create the converted folder. ``mkdir recordings/converted``. Now run ``child-project process . basic standard --format=wav --sampling=16000 --codec=pcm_s16le``.

.. note::

    You may need to change the ``--format`` flag if you're using anything other than .wav files.

Now run ``tree``. You see, we've created the converted/standard folder, which contains the converted recordings. Inside this folder we keep track of the parameters used
to run conversion, and some updated metadata in ``recordings/converted/standard/recordings.csv``. Notably, ``recordings.csv`` remains untouched. These files together
point ChildProject at the converted audio, but also inform it that some audio has in fact been converted, and to always use that in favor of raw audio.

Running models
--------------
.. note::
    A more complete treatment of running models is found `here <https://laac-lscp.github.io/docs/running-models>`_, but this assumes tooling and infrastructure you likely won't have

We will run a few models. These steps can be skipped if they have already be run for you. Technically they are outside the scope of ChildProject, but it
is useful for anyone working with it to know how it is done. The model papers can be found in the references section of the repository README files.

I should warn you, though, this section is by far the most advanced and prone to errors. Hopefully you have someone around to help you run and debug things.

Running VTC
~~~~~~~~~~~
VTC is the Voice Type Classifier, which diarizes our audio into segments with different speakers.

This is not a tutorial on VTC, but it's important that we know how to run it. We will momentarily step out of the dataset. ``cd ..``.

At this point, we follow the steps `here <https://github.com/MarvinLvn/voice-type-classifier/blob/new_model/docs/installation.md>`_ and
`here <https://github.com/MarvinLvn/voice-type-classifier/blob/new_model/docs/applying.md>`_. Make sure `sox <https://sourceforge.net/projects/sox/>`_ is installed.

.. code-block:: bash

    git clone --recurse-submodules https://github.com/MarvinLvn/voice_type_classifier.git
    cd voice_type_classifier
    conda env create -f vtc.yml

Now run ``conda env list``, and you'll see a new environment. Let's use that one while we work with VTC.

.. code-block:: bash

    conda deactivate; conda activate pyannote

The model uses a bash script, ``apply.sh``. Let's run it on what we have

.. code-block:: bash
    ./apply.sh ../dataset-from-scratch/recordings/converted/standard/

.. note::

    This may take a very long time(!!!), extremely long if no gpu is available. I highly, highly recommend running it on a per-file basis.
    For anything above a few hours, and without a gpu in general, it's best to use your institutions' resources.

The command will spit out .rttm annotation files in the ``output_voice_type_classifier`` folder.

Running ALICE
~~~~~~~~~~~~~
Before we get back to our dataset, let's step away from VTC and now run ALICE. ALICE will get unit counts, in particular
phoneme, syllable and word counts over segments derived from VTC.

.. note::

    Fun fact: ALICE actually doing transfer learning on VTC, thus using embeddings derived from the model we just saw earlier

Assuming you're still in the VTC repository folder, step out `cd ..`, and run

.. code-block:: bash

    git clone --recurse-submodules https://github.com/orasanen/ALICE/
    cd ALICE

We also need to make a virtual environment for this model. Run ``conda env create -f [ALICE_Linux.yml | ALICE_macOS.yml]`` depending on your
operating system.

.. note::
    On ARM processors use

    .. code-block:: bash

        CONDA_SUBDIR=osx-64 conda env create -f ALICE_macOS.yml
        conda config --env --set subdir osx-64
    

To activate run ``conda deactivate; conda activate ALICE``. To process your audio files, run either

.. code-block:: bash

    ./run_ALICE.sh ../dataset-from-scratch/recordings/converted/standard/

or

.. code-block:: bash

    ./run_ALICE.sh ../dataset-from-scratch/recordings/converted/standard/ gpu

if you have a gpu available, assuming you have a CUDA-compatible GPU. Note that this will take at least as long as the earlier VTC command.

Running VCM
~~~~~~~~~~~
This vocalisation maturity model lets us estimate occurences of cries, canonical or non-canonical vocalisations.

Assuming you're in the ALICE folder, step out first with `cd ..`. Then run

.. code-block:: bash

    git clone https://github.com/LAAC-LSCP/vcm.git
    cd vcm
    conda create -p ./conda-vcm-env pip python=3.9
    conda deactivate
    conda activate ./conda-vcm-env

For vcm you'll need the SMILExtract (audio feature extractor) binary file, also setting it to executable.

.. code-block:: bash

    wget https://github.com/georgepar/opensmile/blob/master/bin/linux_x64_standalone_static/SMILExtract?raw=true -O SMILExtract
    chmod u+x SMILExtract

(Or instead of wget paste the link into the browser if it didn't work. Remember to chmod as above). To run the model, run

.. code-block:: bash

    ./vcm.sh -a ../dataset-from-scratch/recordings/converted/standard/ -r [path to VTC .rttm output files] -s [path to SMILExtract file] -o ./outputs -j 8

As always, consult with the corresponding GitHub repository if you get stuck, as there is a lot more documentation there.

Adding model outputs
--------------------
Now we want to add our model outputs into our dataset. Depending on the models you've run, you want to create the sets (folders) for the output annotations.

Make sure you're in the root directory of your ``dataset-from-scratch`` dataset. If you were in the VCM/VTC/ALICE folder, go there with ``cd ../dataset-from-scratch``.

Now make the directories for the annotation sets that you'll be working with

.. code-block:: bash

    mkdir -p annotations/vtc/raw
    mkdir -p annotations/vcm/raw
    mkdir -p annotations/alice/raw

And then move the files you have

.. code-block:: bash

    mv [path to dir with vtc output] annotations/vtc/raw
    mv [path to dir with vcm output] annotations/vcm/raw
    mv [path to dir with alice output] annotations/alice/raw

For the rest of the tutorial, we will focus only on vtc annotations,
though vcm, alice and even lena annotations are handled similarly.

Our next aim is to populate our annotations file--we need to import our annotations.

.. code-block:: bash

    child-project automated-import . --set vtc --format vtc_rttm --threads 4

You will likely, like me, get an error saying you need recording durations to be stored. Run ``child-project compute-durations .``. This will
change add a durations column to the recordings metadata. Now run the command above again. Do the same for vcm and alice, changing the ``--set`` and ``--format`` flags accordingly.
You can run ``cat metadata/annotations.csv`` to see that some annotations have been added. We also find that a vtc/converted folder has been created.

Getting Standard Metrics
------------------------
At this point we have enough to get some metrics over our annotations. Run

.. code-block:: bash

    child-project metrics . ACLEW.csv aclew --vtc vtc

Then run ``cat ACLEW.csv`` to inspect the output. Add in the ``--vcm`` and ``--alice`` flags if you have those data available.

Getting Conversational Information
----------------------------------
We have gotten some metrics using the outputted segments from our models. What we can also due is post-process these segments,
and transform them once more into something useful. We have pipelines for that, and one of the most useful one is the conversations pipeline.

.. code-block:: bash

    child-project derive-annotations . conversations --input-set vtc --output-set vtc/conversations

This will create the annotations/vtc/conversations folder with the conversational segmentation.

We can also post-process once more, getting a summary of conversational data

.. code-block:: bash
    
    child-project conversations-summary --set vtc/conversations . conversations.csv standard
