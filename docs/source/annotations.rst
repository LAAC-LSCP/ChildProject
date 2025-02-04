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

Importing annotations to a dataset is taking annotations files in various format, storing them in an annotation
set `raw` folder in the dataset (inside of `annotations/<setname>/raw`) and providing the metadata of the set
inside of the `metannots.yml` file (at `annotations/<setname>/metannots.yml`). Then the importation links those
annotation files to stretches of the recordings of the dataset and creates a standardized csv of the annotations
inside of the converted folder (in `annotations/<setname>/converted`).

For more information on the annotation sets metadata, read its the :ref:`format-annotation-sets-metadata` description.
The annotation set metadata file `metannots.yml` can be created afterwards and does not require new importations
to be taken into account.

Single annotation importation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Annotations can be imported one by one, in bulk or through the automated command. Annotation
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

Automated importation
^^^^^^^^^^^^^^^^^^^^^

The automated method is mostly used for automated annotations. It is made to assume a certain number of parameters on importation, which allows us to perform the usual importations we are doing without additional input. The command will assume the following:
- the annotation files will cover the entirety of the audio they annotate (equivalent to range_onset 0 and range_offset <duration of rec>)
- the annotation files will have timestamps that are not offset compare to the recording (equivalent to time_seek 0)
- the annotation files will be named exactly like the recording they annotate (including the folder they are in) except for the extension, which depends on the format (equivalent to recording_filename = annotation_filename + extension)
- the format used is the same for all the files and needs to be given in the call, it determines the extension for all the annotation files
- the set to import is the same for all files, must be given in the call

.. clidoc::

   child-project automated-import . --help

::

   # import the vtc set by using the vtc_rttm format, all annotation files will need to be with extension ``.rttm``
   child-project automated-import . --set vtc --format vtc_rttm

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

.. _derive-annotations:

Derive annotations
~~~~~~~~~~~~~~~~~~

This command allows to derive a new set of annotations (or adding new lines)
by extracting information from an existing set of annotations. A number of
derivations are available in the package, other derivations can be defined by
the user when using the python api directly.

.. clidoc::

    child-project derive-annotations /path/to/dataset --help

::

    child-project derive-annotations . conversations --input-set vtc --output-set vtc/conversations

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