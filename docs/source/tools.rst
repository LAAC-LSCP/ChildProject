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
