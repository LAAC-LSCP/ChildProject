Managing annotations
--------------------

.. warning::

   You should never run two of the following commands in parallel.
   All of them need to be run sequentially, otherwise the index
   may get corrupted.

   If you need to parallelize the processing to speed it up,
   you can use the ``--threads`` option, which is built-in
   in all of our tools that might require it.


Importation
~~~~~~~~~~~

Single annotation importation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Annotations can be imported one by one or in bulk. Annotation
importation does the following :

1. Convert all input annotations from their original format (e.g. rttm,
   eaf, textgrid..) into the CSV format defined at :ref:`format-input-annotations`
   and stores them into ``annotations/``.
2. Registers them to the annotation index at
   ``metadata/annotations.csv``

Use ``child-project import-annotations`` to import a single annotation.

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
   --right-set alice/output \
   --left-columns speaker_type \
   --right-columns phonemes,syllables,words \
   --output-set alice

Intersect annotations
~~~~~~~~~~~~~~~~~~~~~

In order to combine annotations from different annotators, or to compare them,
it is necessary to calculate which portions of the audio have been annotated by all of them.
This can be done from the command-line interface:

.. clidoc::

    child-project intersect-annotations /path/to/dataset --help

Example:

::

    child-project intersect-annotations /path/to/dataset \
    --sets its textgrid/annotator1 textgrid/annotator2 textgrid/annotator3 \
    --destination intersection.csv

The output dataframe has the same format as the annotations index (see :ref:`format-annotations`).