

.. _format:

Datasets structure
==================

ChildProject assumes your data is structured in a specific way.
This structure is necessary to check, for
instance, that there are no unreferenced files, and no referenced files
that are actually missing. The data curator therefore needs to organize
their data in a specific way (respecting the dataset tree, with all
specified metadata files, and all specified columns within the metadata
files) before their data can be imported.

To be imported, datasets must pass the the validation
routine (see :ref:`tools-data-validation`).
with no error. We also recommend you pay attention to the warnings, and
try to sort as many of those out as possible before submission.

An example of dataset structured according to ChildProject's format
can be found `here <https://gin.g-node.org/LAAC-LSCP/vandam-data>`__.

A set of procedures exists for new datasets, handling the creation of the
base structure and folders as well as linkage to an online repository.
These procedures are used with `datalad <https://www.datalad.org/>`__ and 
can be accessed `here <https://github.com/LAAC-LSCP/datalad-procedures>`__. 
To create a new entire dataset, consider following our `guide in our lab handbook <https://laac-lscp.github.io/docs/create-a-new-dataset.html>`__. 

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
   │   │   │   metannots.yml
   │   │   └───raw
   │   │   │   │   child1.rttm
   │   └───annotator1
   │   │   │   metannots.yml
   │   │   └───raw
   │   │   │   │   child1_3600.TextGrid
   │
   └───docs (*)
   │   │   children.csv
   │   │   recordings.csv
   └───extra
       │   notes.txt

The children and recordings notebooks should be CSV dataframes formatted according to
the standards detailed right below.

   (*) The ``docs`` folder is optional.

.. _format-metadata:

Metadata
--------

Children notebook
~~~~~~~~~~~~~~~~~

The children metadata dataframe needs to be saved at ``metadata/children.csv``.
It should be formatted as instructed below; you can add more fields beyond those that are
standardized, but make sure to document them.

.. index-table:: Children metadata
   :header: children


Recordings notebook
~~~~~~~~~~~~~~~~~~~

The recordings metadata dataframe needs to be saved at
``metadata/recordings.csv``.
It should be formatted as instructed below; you can add more fields beyond those that are
standardized, but make sure to document them.

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

This is also useful if your metadata includes many columns and you'd like to
spread it across several dataframes. This can also be used to deliver survey data
in a separate file.

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
   For instance, if one dataframe is located at `metadata/children/dates-of-birth.csv` ,
   an error will be thrown if another dataframe exists at
   `metadata/children/private/dates-of-birth.csv` .

Annotations
-----------

Upon importation, annotations are converted to standardized
CSV dataframes (using built-in or custom ingestors)
and registered into an index.
The index of annotations stores the list of each interval
that has been annotated for each annotator.
This allows a number of functionalities
such as the quick computation of the intersection of the
portions of audio covered by a given set of annotators.

.. _format-annotation-sets-metadata:

Annotation sets metadata
~~~~~~~~~~~~~~~~~~~~~~~~

Additionally, information about a set of annotations (an ensemble grouping annotations from the same source)
must be stored inside the set folder in a file named `metannots.yml`. This is a yaml formatted text file with
a combination of key => values fields defined in it. This file is not mandatory to have but it is veery
strongly encouraged to create it whenever adding a set of annotations.

Here is an example of the content of `metannots.yml` for the annotations from an automated tool (VTC):

.. code-block:: YAML

    segmentation: 'vtc'
    segmentation_type: 'permissive'
    method: 'automated'
    annotation_algorithm_name: 'VTC'
    annotation_algorithm_publication: 'Lavechin, M., Bousbib, R., Bredin, H., Dupoux, E., & Cristia, A. (2020). An open-source voice type classifier for child-centered daylong recordings. Interspeech. Online open access: https://www.isca-archive.org/interspeech_2020/lavechin20_interspeech.pdf'
    annotation_algorithm_version: '1'
    annotation_algorithm_repo: 'https://github.com/MarvinLvn/voice-type-classifier/tree/e443d8cfc40f7076eea903958d9344d4aa427cc2'
    date_annotation: '2024-04-07'
    has_speaker_type: 'Y'


And another example for a human annotated set of annotations:

.. code-block:: YAML

    segmentation: 'textgrid2'
    segmentation_type: 'permissive'
    method: 'manual'
    sampling_method: 'high-volubility'
    sampling_target: 'fem'
    sampling_count: 17
    sampling_unit_duration: 50000
    recording_selection: 'all recordings'
    participant_selection: '1 to 2 yo'
    annotator_name: 'Ivan Cliao'
    annotator_experience: 5
    date_annotation: '2019-07-16'
    has_speaker_type: 'Y'
    has_transcription: 'Y'
    has_vcm_type: 'Y'
    has_addressee: 'N'


All the supported fields are listed below with their description. Custom fields may be used but won't be checked.

.. index-table:: Metadata fields supported for annotation sets
    :header: sets-metadata

.. _format-annotations-segments:

Annotations format
~~~~~~~~~~~~~~~~~~

The package provides functions to convert any annotation into the
following CSV format, with one row per segment (e.g. per vocalization event):

.. index-table:: Annotations format
   :header: annotation_segments

Custom columns may be used, although they should be documented somewhere in your dataset.

.. _format-annotations:

Annotations index
~~~~~~~~~~~~~~~~~

.. warning::

    The index is maintained through the package functions only; it should never be updated by hand.

Annotations are indexed in one unique dataframe located at
``/metadata/annotations.csv`` , with the following format :

.. index-table:: Annotations metadata
   :header: annotations

Below is shown an example of an index file
(some uninformative columns were hidden for clarity).
In this case, one recording has been fully
annotated using the Voice Type Classifier (vtc),
and partially annotated by two humans (LM and SP).
These humans have both annotated the same seven 15 second clips.

.. csv-table:: 
   :header-rows: 1

   set,recording_filename,time_seek,range_onset,range_offset,raw_filename,format,annotation_filename
   vtc,A730/A730_001105.wav,0,0,42764250,A730/A730_001105.rttm,vtc_rttm,A730/A730_001105_0_42764250.csv
   eaf_2021/SP,A730/A730_001105.wav,0,2910000,2925000,A730_001105.eaf,eaf,A730/A730_001105_2910000_2925000.csv
   eaf_2021/SP,A730/A730_001105.wav,0,4680000,4695000,A730_001105.eaf,eaf,A730/A730_001105_4680000_4695000.csv
   eaf_2021/SP,A730/A730_001105.wav,0,4695000,4710000,A730_001105.eaf,eaf,A730/A730_001105_4695000_4710000.csv
   eaf_2021/SP,A730/A730_001105.wav,0,14055000,14070000,A730_001105.eaf,eaf,A730/A730_001105_14055000_14070000.csv
   eaf_2021/SP,A730/A730_001105.wav,0,15030000,15045000,A730_001105.eaf,eaf,A730/A730_001105_15030000_15045000.csv
   eaf_2021/SP,A730/A730_001105.wav,0,36465000,36480000,A730_001105.eaf,eaf,A730/A730_001105_36465000_36480000.csv
   eaf_2021/SP,A730/A730_001105.wav,0,39450000,39465000,A730_001105.eaf,eaf,A730/A730_001105_39450000_39465000.csv
   eaf_2021/LM,A730/A730_001105.wav,0,2910000,2925000,A730_001105.eaf,eaf,A730/A730_001105_2910000_2925000.csv
   eaf_2021/LM,A730/A730_001105.wav,0,4680000,4695000,A730_001105.eaf,eaf,A730/A730_001105_4680000_4695000.csv
   eaf_2021/LM,A730/A730_001105.wav,0,4695000,4710000,A730_001105.eaf,eaf,A730/A730_001105_4695000_4710000.csv
   eaf_2021/LM,A730/A730_001105.wav,0,14055000,14070000,A730_001105.eaf,eaf,A730/A730_001105_14055000_14070000.csv
   eaf_2021/LM,A730/A730_001105.wav,0,15030000,15045000,A730_001105.eaf,eaf,A730/A730_001105_15030000_15045000.csv
   eaf_2021/LM,A730/A730_001105.wav,0,36465000,36480000,A730_001105.eaf,eaf,A730/A730_001105_36465000_36480000.csv
   eaf_2021/LM,A730/A730_001105.wav,0,39450000,39465000,A730_001105.eaf,eaf,A730/A730_001105_39450000_39465000.csv

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

Documentation
-------------

An important aspect of a dataset is its documentation.
Documentation includes:

 - authorship, references, contact information
 - a description of the corpus (population, collection process, etc.)
 - instructions to re-use the data
 - description of the data itself (e.g. a definition of each metadata field)

We currently do not provide a format for *all* these annotations.
It is up to you to decide how to provide users with each of these information.

However, we suggest several options below.

Metadata and annotations
~~~~~~~~~~~~~~~~~~~~~~~~

The ChildProject package supports a machine-readable format 
to describe the contents of the metadata and the annotations.

This format consists in CSV dataframe structured according 
to the following table:

.. index-table:: Machine-readable documentation
   :header: documentation

- Documentation for the children metadata should be stored in ``docs/children.csv``
- Documentation for the recordings metadata should be stored in ``docs/recordings.csv``
- Documentation for annotations should be stored in ``docs/annotations.csv``

Authorship
~~~~~~~~~~

We recommend DataCite's .yaml format (see `here <https://github.com/G-Node/gogs/blob/master/conf/datacite/datacite.yml>`__)

