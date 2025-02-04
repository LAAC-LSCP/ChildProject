Basic tools
===========

Dataset initialization
----------------------

This command allows you to create a new and empty dataset, with the correct structure

.. clidoc::

   child-project init /path/to/dataset --help

Example:

.. code:: bash

   # create a dataset in a folder named mydataset
   child-project init mydataset

.. _tools-data-validation:

Data validation
---------------

This is typically done (repeatedly!) in the process of importing your
data into our format for the first time, but you should also do this
whenever you make a change to the dataset.

Looks for errors and inconsistency in the metadata, or for missing
audios. The validation will pass if formatting instructions are met
(see :ref:`format`).

.. clidoc::

   child-project validate /path/to/dataset --help

Example:

.. code:: bash

   # validate the metadata and raw recordings
   child-project validate /path/to/dataset

   # validate the metadata only
   child-project validate /path/to/dataset --ignore-recordings 

   # validate the metadata and the recordings of the 'standard' profile
   # (in recordings/converted/standard)
   child-project validate /path/to/dataset --profile standard 

   # validate the metadata and all annotations within /path/to/dataset/annotations
   child-project validate /path/to/dataset --ignore-recordings --annotations /path/to/dataset/annotations/*

   # validate the metadata and annotations from the 'textgrid' set
   child-project validate /path/to/dataset --ignore-recordings --annotations /path/to/dataset/annotations/textgrid/*

Dataset overview
----------------

An overview of the contents of a dataset can be obtained with the
``child-project overview`` command.

.. clidoc::

   child-project overview --help

Example:

.. clidoc::

    child-project overview ../examples/valid_raw_data

Annotation sets metadata
------------------------

An overview of the annotation sets of a dataset can be obtained with the
``child-project sets-metadata`` command. The output can be formatted to
be human readable or parsable (csv).

.. clidoc::

   child-project sets-metadata --help

Example:

.. clidoc::

    child-project sets-metadata ../examples/valid_raw_data

Compute recordings duration
---------------------------

Compute recordings duration in ms and store in into a column named ‘duration’
in the metadata.

.. clidoc::

   child-project compute-durations /path/to/dataset --help

Compute the correlation between audio files
-------------------------------------------

Compute the correlation between two audio files and prints a divergence score.
The divergence is computed over a given duration (default 5min) that can be changed with the `--interval` option.
One segment of that duration is taken randomly, the difference in audio signal is calculated and averaged over the total duration. The result is printed as a divergence score.
The closer the score is to 0, the more likely it is the 2 files are identical. We can consider that scores below 0.1 reflect a very high probability that the files are the same. At the other end of the spectrum, values higher than 1 almost certainly means they are different recordings.
So a window exists in which we can't be sure and would need additional correlation computations or manual checks. Running the correlation multiple time is useful because files that are different have a high variability in score whereas similar files will have a much more consistent output.

Giving a higher `--interval` value may take more time to compute.

.. clidoc::

   child-project compare-recordings /path/to/dataset --help