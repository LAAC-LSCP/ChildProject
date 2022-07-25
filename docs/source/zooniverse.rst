.. _zooniverse:

Zooniverse
==========

Introduction
~~~~~~~~~~~~

We are providing here a pipeline to create, upload and analyse long
format recordings using the Zooniverse citizen science platform.

We have an open project aimed at adding vocal maturity labels to
segments LENA labeled as being key child in Zooniverse
(https://www.zooniverse.org/projects/chiarasemenzin/maturity-of-baby-sounds).

If you would like your data labeled with this project, here is what
you’d need to do.

1. Get in touch with us, so we know you are interested!
2. Have someone trustworthy & with some coding skills (henceforth, the RA) create a database using the formatting instructions (see :ref:`format`).
3. Have the RA create an account on Zooniverse (top right of zooniverse.org) for them and yourself, & provide us with both handles. The RA should first update the team section to add you (have ready a picture and a blurb). The RA can also add your institution’s logo if you’d like. Both of these are done in the `lab section <https://www.zooniverse.org/lab/10073>`__.
4. The RA will then follow the instructions in the present README to create subjects and push up your data – see below.
5. We also ask the RA to pitch in and help answer questions in the `forum <https://www.zooniverse.org/projects/chiarasemenzin/maturity-of-baby-sounds/talk>`__, at least one comment a day.
6. You can visit the `stats section <https://www.zooniverse.org/projects/chiarasemenzin/maturity-of-baby-sounds/stats>`__ to look at how many annotations are being done.

You can also use this code and your own knowledge to set up a new
project of your own.

.. note::

   This is the documentation for the Zooniverse pipeline of the package.
   We also provide separately a more detailed `step-by-step tutorial <https://gin.g-node.org/LAAC-LSCP/zoo-campaign>`__ for creating a campaign of classification using Zooniverse and ChildProject.

Overview
~~~~~~~~

.. clidoc::

   child-project zooniverse --help


Chunk extraction
~~~~~~~~~~~~~~~~

The ``extract-chunks`` pipeline creates wav and mp3 files for each chunk of audio to be classified on Zooniverse.
It also saves a record of all these chunks into a CSV dataframe.
This record can then be provided to the ``upload-chunks`` command, in order to upload
the chunks to zooniverse.

.. note::

    ``extract-chunks`` will require the list of segments to classify, which are provided as a CSV dataframe with three columns:
    ``recording_filename``, ``segment_onset``, and ``segment_offset``. The path to this dataframe has to be specified with the
    ``--segments`` parameter. 
    
    The list of segments can be generated with any of the samplers we provide (see :ref:`samplers`), but custom lists 
    may also be provided.

Optionally, the segments provided to the pipeline can be split into chunks of the desired duration.
By setting this duration to sufficently low values (e.g. 500 milliseconds), one can ensure that
no meaningful information could be recovered while listening to the audio on Zooniverse.
This is useful when the segments of audio provided to the pipeline may contain confidential information.

.. clidoc::

   child-project zooniverse extract-chunks /path/to/dataset --help

If it does not exist, DESTINATION is created. Audio chunks are saved in
wav and mp3 in ``DESTINATION/chunks``. Metadata is stored in a CSV file
into ``DESTINATION/``.

The output dataframe will contain the following columns:

.. csv-table:: 
   
   index,"self-generated integer index"
   recording_filename,"recording from which the chunk as extracted"
   onset,"onset timestamp of the chunk within the recording"
   offset,"offset timestamp of the chunk within the recording"
   segment_onset,"onset timestamp of the segment from which the chunk was extracted"
   segment_offset,"offset timestamp of the segment from which the chunk was extracted"
   wav,"name of the wav file"
   mp3,"name of the mp3 file"
   png,"name of the png file if spectrogram option"
   date_extracted,"date at which the chunk was extracted"
   uploaded,"boolean flag set to True if the chunk was uploaded to Zooniverse, False otherwise"
   project_id,"zooniverse project ID"
   subject_set,"name of the Zooniverse subject set"
   zooniverse_id,"subject's Zooniverse ID"
   keyword,"custom keyword provided by the user to label the chunks"

Chunk upload
~~~~~~~~~~~~

Once the chunks have been extracted, the next step is to upload them to Zooniverse.
Note that due to quotas, it is recommended to upload only a few at time (e.g. 1000 per day).

You will need to provide the numerical id of your Zooniverse project, as well as your Zooniverse credentials.

``child-project zooniverse upload-chunks`` uploads as many batches of audio chunks as specified to Zooniverse, and
updates the chunks metadata accordingly, by setting the `zooniverse_id` field and `uploaded` to `True`.

.. clidoc::

   child-project zooniverse upload-chunks /path/to/dataset --help


Classifications retrieval
~~~~~~~~~~~~~~~~~~~~~~~~~

.. clidoc::

   child-project zooniverse retrieve-classifications /path/to/dataset --help

Retrieve classifications and save them as ``DESTINATION``.
The optional ``--chunks`` parameter can be used to match the classifications with the chunks metadata. Only the classifications
that match the metadata will be saved.

.. warning::
   Retrieving chunks may take a long time for large projects.