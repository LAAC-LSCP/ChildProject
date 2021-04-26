

.. _format:

Datasets structure
==================

ChildProject assumes your data is structured in a specific way
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

Children notebook
~~~~~~~~~~~~~~~~~

The children metadata dataframe needs to be saved at ``metadata/children.csv``.

.. index-table:: Children metadata
   :header: children


Recordings notebook
~~~~~~~~~~~~~~~~~~~

The recordings metadata dataframe needs to be saved at
``metadata/recordings.csv``.

.. index-table:: Recordings metadata
   :header: recordings

Splitting the metadata across several files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes, access to parts of the metadata should be limited
to a list of authorized users. This can be achieved by moving confidential
information out of the main notebook to a separate CSV file to
be only delivered to authorized users. These additional files
should be placed according to the table below:


.. csv-table:: Additional metadata
   :header: data,main notebook,location of additional notebooks

   children,``metadata/children.csv``,``metadata/children/``
   recordings,``metadata/recordings.csv``,``metadata/recordings/``

There can be as many additional notebooks as necessary, and recursion
is permitted.

.. note::

   In case two or more notebooks contain the same column, the files
   whose names come first in alphabetical order will prevail while
   loading the dataset with our package. For instance, if
   ``child_dob`` is specified in both  ``metadata/recordings/0_private.csv``
   and ``metadata/recordings/1_public.csv``, the values in the former file will prevail if it is available.
   This is useful when anonymized values for a certain parameter still need to be shared,
   but should be replaced with the true values for those who have access to the full dataset.

.. warning::

   For recursive metadata, two dataframes cannot share the same basename.
   For instance, if one dataframe is located at `metadata/children/dates-of-birth.csv`,
   an error will be thrown if another dataframe exists at
   `metadata/children/private/dates-of-birth.csv `.

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

.. note::
   In order to avoid rounding errors, all timestamps are integers,
   expressed in milliseconds.

