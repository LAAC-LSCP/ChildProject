Basic tools
===========

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

::

   child-project validate /path/to/dataset

Convert recordings
------------------

Converts all recordings in a dataset to a given encoding. Converted
audios are stored into ``recordings/converted/<profile-name>``.

.. clidoc::

   child-project convert /path/to/dataset --help

Example:

::

   child-project convert /path/to/dataset --name=16kHz --format=wav --sampling=16000 --codec=pcm_s16le

We typically run the following, to split long sound files every 15
hours, because the software we use for human annotation (ELAN, Praat)
works better with audio that is maximally 15h long:

::

   child-project convert /path/to/dataset --name=16kHz --split=15:00:00 --format=wav --sampling=16000 --codec=pcm_s16le

Multi-core audio conversion with slurm on a cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have access to a cluster with slurm, you can use a command like
the one below to batch-convert your recordings. Please note that you may
need to change some details depending on your cluster (eg cpus per
task). If needed, refer to the `slurm user
guide <https://slurm.schedmd.com/quickstart.html>`__

::

   sbatch --mem=64G --time=5:00:00 --cpus-per-task=4 --ntasks=1 -o namibia.txt child-project convert /path/to/dataset --name standard --format WAV --codec pcm_s16le --sampling 16000 --threads 4`

Compute recordings duration
---------------------------

Compute recordings duration and store in into a column named ‘duration’
in the metadata.

.. clidoc::

   child-project compute-durations /path/to/dataset --help

Managing annotations
--------------------

Importation
~~~~~~~~~~~

Single file importation
^^^^^^^^^^^^^^^^^^^^^^^

Annotations can be imported one by one or in bulk. Annotation
importation does the following :

1. Convert all input annotations from their original format (e.g. rttm,
   eaf, textgrid..) into the CSV format defined at :ref:`format-input-annotations`
   and stores them into ``annotations/``.
2. Registers them to the annotation index at
   ``metadata/annotations.csv``



Use ``child-project import-annotations`` to import a single annotation
file.

.. clidoc::

   child-project import-annotations /path/to/dataset --help

Example:

::

   child-project import-annotations /path/to/dataset \
      --set eaf \
      --recording_filename sound.wav \
      --time_seek 0 \
      --raw_filename example.eaf \
      --range_onset 0 \
      --range_offset 300 \
      --format eaf

Find more information about the allowed values for each parameter, see :ref:`format-input-annotations`.

.. _tools-annotations-bulk-importation:

Bulk importation
^^^^^^^^^^^^^^^^

Use this to do bulk importation of many annotation files.

::

   child-project import-annotations /path/to/dataset --annotations /path/to/dataframe.csv

The input dataframe ``/path/to/dataframe.csv`` must have one entry per
annotation to import, according to the format specified at :ref:`format-input-annotations`.


Rename a set of annotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rename a set of annotations. This will move the annotations themselves,
and update the index (``metadata/annotations.csv``) accordingly.

.. clidoc::

   child-project rename-annotations /path/to/dataset --help

Example:

::

   child-project rename-annotations /path/to/dataset --set vtc --new-set vtc_1

Remove a set of annotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This will deleted converted annotations associated to a given set and
remove them from the index.

.. clidoc::

   child-project remove-annotations /path/to/dataset --help

::

   child-project remove-annotations /path/to/dataset --set vtc

ITS annotations anonymization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LENA .its files might contain information that can help recover the identity of the participants, which may be undesired.
This command anonymizes .its files, based on a routine by `HomeBank
<https://github.com/HomeBankCode/ITS_annonymizer>`_.

.. clidoc::

   child-project anonymize /path/to/dataset --help

::

   child-project anonymize /path/to/dataset --input-set lena --output-set lena/anonymous

Merge annotation sets
~~~~~~~~~~~~~~~~~~~~~

Some processing tools use pre-existing annotations as an input,
and label the original segments with more information. This is
typically the case of ALICE, which labels segments generated
by the VTC. In this case, one might want to merge the ALICE
and VTC annotations altogether. This can be done with ``child-project merge-annotations``.

.. clidoc::

   child-project merge-annotations /path/to/dataset --help

::

   child-project merge-annotations /path/to/dataset \
   --left-set vtc \
   --right-set alice \
   --left-columns speaker_id,ling_type,speaker_type,vcm_type,lex_type,mwu_type,addresseee,transcription \
   --right-columns phonemes,syllables,words \
   --output-set alice_vtc