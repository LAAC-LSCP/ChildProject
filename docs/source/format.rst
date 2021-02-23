

.. _format:

Datasets structure
==================

ChildRecordsData assumes your data is structured in a specific way
before it is imported. This structure is necessary to check, for
instance, that there are no unreferenced files, and no referenced files
that are actually missing. The data curator therefore needs to organize
their data in a specific way (respecting the dataset tree, with all
specified metadata files, and all specified columns within the metadata
files) before their data can be imported.

To be imported, datasets must pass the the validation
routine (see :ref:`tools-data-validation`).
with no error. We also recommend you pay attention to the warnings, and
try to sort as many of those out as possible before submission.

Dataset tree
------------

All datasets should have this structure before import (so you need to
organize your files into this structure):

::

   project
   │   
   │
   └───metadata
   │   │   children.csv
   │   │   recordings.csv
   │   │   annotations.csv
   |
   └───recordings
   │   └───raw
   │   │   │   recording1.wav
   │
   └───annotations
   │   └───vtc
   │   │   └───raw
   │   │   │   │   child1.rttm
   │   └───annotator1
   │   │   └───raw
   │   │   │   │   child1_3600.TextGrid
   │
   └───extra
       │   notes.txt

The children and recordings notebooks should be formatted according to
the standards detailed right below.

.. _format-metadata:

Metadata
--------

children notebook
~~~~~~~~~~~~~~~~~

The children dataframe needs to be saved at ``metadata/children.csv``.

.. index-table:: Children metadata
   :header: children

recording notebook
~~~~~~~~~~~~~~~~~~

The recordings dataframe needs to be saved at
``metadata/recordings.csv``.

.. index-table:: Recordings metadata
   :header: recordings

Annotations
-----------

.. _format-annotations-segments:

Annotations format
~~~~~~~~~~~~~~~~~~

The package provides functions to convert any annotation into the
following csv format, with one row per segment :

.. index-table:: Annotations format
   :header: annotation_segments

.. _format-annotations:

Annotations index
~~~~~~~~~~~~~~~~~

Annotations are indexed in one unique dataframe located at
``/metadata/annotations.csv``, with the following format :

.. index-table:: Annotations metadata
   :header: annotations

.. _format-input-annotations:

Annotation importation input format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The annotations importation script (:ref:`tools-annotations-bulk-importation`) and python method (:meth:`ChildProject.annotations.AnnotationManager.import_annotations`) take a dataframe of the
following format as an input:

.. index-table:: Input annotations
   :header: input_annotations