Basic tools
===========

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

.. code:: bash

   $ child-project overview .

   recordings:
   lena: 288.00 hours, 0/18 files locally available
   olympus: 49.57 hours, 0/3 files locally available
   usb: 223.42 hours, 0/20 files locally available
   
   annotations:
   alice: 560.99 hours, 0/40 files locally available
   alice_vtc: 560.99 hours, 0/40 files locally available
   eaf/nk: 1.47 hours, 0/88 files locally available
   lena: 272.00 hours, 0/17 files locally available
   textgrid/mm: 8.75 hours, 0/525 files locally available
   vtc: 560.99 hours, 40/40 files locally available

Compute recordings duration
---------------------------

Compute recordings duration in ms and store in into a column named ‘duration’
in the metadata.

.. clidoc::

   child-project compute-durations /path/to/dataset --help

Compute the correlation between audio files
---------------------------

Compute the correlation between two audio files and prints a similarity score.
The similarity is evaluated over a given duration (default 5min) that can be changed with the `--interval` option.
One segment of that duration taken randomly in the recordings total duration is evluated and a similarity score is computed for every frame of that segment. Then only the highest score is kept (meaning the frame where the 2 files differed the most) and is printed.
The closer the score is to 0, the more likely it is the 2 files are identical. We can consider that scores below <VALUE>(0.01?) reflect a very high probability that the files are the same. At the other end of the spectrum, values higher than <VALUE>(0?) almost certainly means they are different.
So a window exists in which we can't be sure and would need additional correlation computations or manual checks.

Giving a higher `-- interval` value may take more time to compute and will yield higher scores on average just because the sample is bigger, but it should be more reliable.

.. clidoc::

   child-project correlate-audio /path/to/dataset --help