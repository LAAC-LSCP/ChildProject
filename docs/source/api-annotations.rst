Annotations
===========

Annotations can be managed through both the command-line interface and the python
API. This section documents the principle features of the API for the management
of annotations.

.. note:: 

    In order to reproduce the following examples, you will need to install
    the public VanDam corpus and its annotations using datalad:
    
    .. code-block:: bash
    
        datalad install git@gin.g-node.org:/LAAC-LSCP/vandam-data.git
        datalad get vandam-data/annotations


Reading annotations
~~~~~~~~~~~~~~~~~~~

Annotations are managed with :class:`ChildProject.annotations.AnnotationManager` class.
The first step is create an instance of it based on the target project.

The :meth:`~ChildProject.annotations.AnnotationManager.read` method reads the index of annotations
from ``metadata/annotations.csv`` and stores into its
:attr:`~ChildProject.annotations.AnnotationManager.annotations` attribute:


.. code-block:: python

    >>> from ChildProject.projects import ChildProject
    >>> from ChildProject.annotations import AnnotationManager
    >>> project = ChildProject('vandam-data')
    >>> am = AnnotationManager(project)
    >>> am.read()
    ([], ["vandam-data/metadata/annotations.csv: 'chat' is not a permitted value for column 'format' on line 4, should be any of [csv,vtc_rttm,vcm_rttm,alice,its,TextGrid,eaf,cha,NA]"])
    >>> am.annotations
                set recording_filename  time_seek  range_onset  range_offset             raw_filename    format  filter                annotation_filename          imported_at  error package_version
    2           its    BN32_010007.mp3          0            0      50464512          BN32_010007.its       its     NaN                BN32_010007_0_0.csv  2021-03-06 22:55:06    NaN           0.0.1
    3           vtc    BN32_010007.mp3          0            0      50464512         BN32_010007.rttm  vtc_rttm     NaN                BN32_010007_0_0.csv  2021-05-12 19:28:25    NaN           0.0.1
    4           cha    BN32_010007.mp3          0            0      50464512          BN32_010007.cha      chat     NaN                BN32_010007_0_0.csv  2021-05-12 19:39:05    NaN           0.0.1
    5           eaf    BN32_010007.mp3          0      4138389       4199976          BN32_010007.eaf       eaf     NaN    BN32_010007_4138389_4199976.csv  2021-07-14 17:39:50    NaN           0.0.1
    6           eaf    BN32_010007.mp3          0      4438842       4499995          BN32_010007.eaf       eaf     NaN    BN32_010007_4438842_4499995.csv  2021-07-14 17:39:50    NaN           0.0.1
    7           eaf    BN32_010007.mp3          0     13199449      13256801          BN32_010007.eaf       eaf     NaN  BN32_010007_13199449_13256801.csv  2021-07-14 17:39:50    NaN           0.0.1
    8           eaf    BN32_010007.mp3          0     37496002      37558424          BN32_010007.eaf       eaf     NaN  BN32_010007_37496002_37558424.csv  2021-07-14 17:39:50    NaN           0.0.1
    9           eaf    BN32_010007.mp3          0     37616206      37679577          BN32_010007.eaf       eaf     NaN  BN32_010007_37616206_37679577.csv  2021-07-14 17:39:50    NaN           0.0.1
    10  cha/aligned    BN32_010007.mp3          0            0      47725356  BN32_010007-aligned.csv       csv     NaN         BN32_010007_0_47725356.csv  2021-07-15 16:15:48    NaN           0.0.1

As seen in this example, :attr:`~ChildProject.annotations.AnnotationManager.annotations` only
contains the index of annotations, not their contents. To retrieve the actual annotations,
use :meth:`~ChildProject.annotations.AnnotationManager.get_segments`:

.. code-block:: python

    >>> selection = am.annotations[am.annotations['set'].isin(['cha', 'vtc'])]
    >>> segments = am.get_segments(selection)
    >>> segments

        segment_onset  segment_offset speaker_type      raw_filename  set  annotation_filename participant  ... range_onset range_offset    format filter          imported_at error package_version
    0               9992           10839       SPEECH  BN32_010007.rttm  vtc  BN32_010007_0_0.csv         NaN  ...           0     50464512  vtc_rttm    NaN  2021-05-12 19:28:25   NaN           0.0.1
    1              10004           10814          CHI  BN32_010007.rttm  vtc  BN32_010007_0_0.csv         NaN  ...           0     50464512  vtc_rttm    NaN  2021-05-12 19:28:25   NaN           0.0.1
    2              11298           11953       SPEECH  BN32_010007.rttm  vtc  BN32_010007_0_0.csv         NaN  ...           0     50464512  vtc_rttm    NaN  2021-05-12 19:28:25   NaN           0.0.1
    3              11345           11828          CHI  BN32_010007.rttm  vtc  BN32_010007_0_0.csv         NaN  ...           0     50464512  vtc_rttm    NaN  2021-05-12 19:28:25   NaN           0.0.1
    4              12113           12749          FEM  BN32_010007.rttm  vtc  BN32_010007_0_0.csv         NaN  ...           0     50464512  vtc_rttm    NaN  2021-05-12 19:28:25   NaN           0.0.1
                ...             ...          ...               ...  ...                  ...         ...  ...         ...          ...       ...    ...                  ...   ...             ...
    31875       49705416        49952432          CHI   BN32_010007.cha  cha  BN32_010007_0_0.csv         CHI  ...           0     50464512      chat    NaN  2021-05-12 19:39:05   NaN           0.0.1
    31876       49952432        50057166          CHI   BN32_010007.cha  cha  BN32_010007_0_0.csv         CHI  ...           0     50464512      chat    NaN  2021-05-12 19:39:05   NaN           0.0.1
    31877       50057166        50173260          CHI   BN32_010007.cha  cha  BN32_010007_0_0.csv         CHI  ...           0     50464512      chat    NaN  2021-05-12 19:39:05   NaN           0.0.1
    31878       50173260        50330885          CHI   BN32_010007.cha  cha  BN32_010007_0_0.csv         CHI  ...           0     50464512      chat    NaN  2021-05-12 19:39:05   NaN           0.0.1
    31879       50330885        50397134          CHI   BN32_010007.cha  cha  BN32_010007_0_0.csv         CHI  ...           0     50464512      chat    NaN  2021-05-12 19:39:05   NaN           0.0.1

    [31880 rows x 22 columns]
    
.. warning::

    Trying to load all annotations at once may quickly lead to out-of-memory errors,
    especially with automated annotators convering thousands of hours of audio.
    Memory issues can be alleviated by processing the data sequentially, e.g.
    by treating one recording after another.

Importing annotations
~~~~~~~~~~~~~~~~~~~~~

Although importing annotations can be done using the command-line tool, 
sometimes it is more efficient to do it directly with the python API; it can
even become necessary when custom converters (the functions that transform
any kind of annotations into the CSV format used by the package) need to be used.

Two examples are given below (one using built-in converters, one using a custom converter).
In order to reproduce them, please make a copy of the original annotations:

.. code:: bash

    mkdir vandam-data/annotations/playground
    cp -r vandam-data/annotations/its vandam-data/annotations/playground

Built-in formats
----------------

The following code imports only the annotations from the LENA that
correspond to the second hour of the audio. The package natively
supports LENA's .its annotations.

Annotations are imported using :meth:`~ChildProject.annotations.AnnotationManager.import_annotations`.
This first input argument of this method must be a pandas dataframe of all the annotations that need to 
be imported. This dataframe should be structured according to the format defined at :ref:`format-input-annotations`.

.. code-block:: python

    >>> import pandas as pd
    >>> input = pd.DataFrame([{
    ...     'set': 'playground/its',
    ...     'recording_filename': 'BN32_010007.mp3',
    ...     'time_seek': 0,
    ...     'range_onset': 3600*1000,
    ...     'range_offset': 7200*1000,
    ...     'raw_filename': 'BN32_010007.its',
    ...     'format': 'its'
    ... }])
    >>> am.import_annotations(input, threads = 1)
                set recording_filename  time_seek  range_onset  range_offset     raw_filename format        annotation_filename          imported_at package_version
    0  playground/its    BN32_010007.mp3          0      3600000       7200000  BN32_010007.its    its  BN32_010007_0_3600000.csv  2021-05-12 20:37:43           0.0.1


After reloading the index of annotations, the newly inserted entry now appears:

.. code-block:: python

    >>> am.read()
    ([], [])
    >>> am.annotations
                set recording_filename  time_seek  range_onset  range_offset      raw_filename    format  filter        annotation_filename          imported_at  error package_version
    2             its    BN32_010007.mp3          0            0      50464512   BN32_010007.its       its     NaN        BN32_010007_0_0.csv  2021-03-06 22:55:06    NaN           0.0.1
    3             vtc    BN32_010007.mp3          0            0      50464512  BN32_010007.rttm  vtc_rttm     NaN        BN32_010007_0_0.csv  2021-05-12 19:28:25    NaN           0.0.1
    4             cha    BN32_010007.mp3          0            0      50464512   BN32_010007.cha      chat     NaN        BN32_010007_0_0.csv  2021-05-12 19:39:05    NaN           0.0.1
    5  playground/its    BN32_010007.mp3          0      3600000       7200000   BN32_010007.its       its     NaN  BN32_010007_0_3600000.csv  2021-05-12 20:37:43    NaN           0.0.1

Built-in converters include: LENA's its, VTC's and VCM's rttms, ALICE, ACLEW DAS eaf files.
To import annotations under other formats, custom converters are needed.

Custom converter
----------------

A converter is a function that takes a filename for only input, and return a dataframe
complying with the specifications defined in :ref:`format-annotations-segments`.

The output dataframe _must_ contain at least a ``segment_onset`` and a ``segment_offset`` columns
expressing the onset and offset of each segment in milliseconds as
integers.

You are free to add as many extra columns as you want. It is however preferable to follow the
standards listed in :ref:`format-annotations-segments` when possible.

In our case, we'll write a very simple converter to extract only the segments onset and offset
from rttm files:

.. code-block:: python

    >>> def convert_rttm(filename: str):
    ...     df = pd.read_csv(filename, sep = " ", names = ['type', 'file', 'chnl', 'tbeg', 'tdur', 'ortho', 'stype', 'name', 'conf', 'unk'])
    ...     df['segment_onset'] = df['tbeg'].mul(1000).round().astype(int)
    ...     df['segment_offset'] = (df['tbeg']+df['tdur']).mul(1000).round().astype(int)
    ...     df.drop(['type', 'file', 'chnl', 'tbeg', 'tdur', 'ortho', 'stype', 'name', 'conf', 'unk'], axis = 1, inplace = True)
    ...     return df
    ... 
    >>> 

The converter can now be used with :meth:`~ChildProject.annotations.AnnotationManager.import_annotations`:

.. code-block:: python

    >>> input = pd.DataFrame([{
    ...     'set': 'playground/vtc',
    ...     'recording_filename': 'BN32_010007.mp3',
    ...     'time_seek': 0,
    ...     'range_onset': 3600*1000,
    ...     'range_offset': 7200*1000,
    ...     'raw_filename': 'BN32_010007.rttm',
    ...     'format': 'custom_rttm'
    ... }])
    >>> am.import_annotations(input, threads = 1, import_function = convert_rttm)
                set recording_filename  time_seek  range_onset  range_offset      raw_filename       format        annotation_filename          imported_at package_version
    0  playground/vtc    BN32_010007.mp3          0      3600000       7200000  BN32_010007.rttm  custom_rttm  BN32_010007_0_3600000.csv  2021-05-13 17:25:20           0.0.1

The contents of the output CSV file can be checked:

.. code-block:: python

    >>> rttm = pd.read_csv('vandam-data/annotations/playground/vtc/converted/BN32_010007_0_3600000.csv')
    >>> rttm
        segment_onset  segment_offset      raw_filename
    0           3600401         3601370  BN32_010007.rttm
    1           3600403         3601464  BN32_010007.rttm
    2           3601503         3602843  BN32_010007.rttm
    3           3601527         3602833  BN32_010007.rttm
    4           3604075         3605570  BN32_010007.rttm
    ...             ...             ...               ...
    1622        7010992         7011243  BN32_010007.rttm
    1623        7011495         7011615  BN32_010007.rttm
    1624        7033826         7034142  BN32_010007.rttm
    1625        7036539         7037008  BN32_010007.rttm
    1626        7036556         7036996  BN32_010007.rttm

    [1627 rows x 3 columns]

.. warning::

    Do not import the same file twice, as duplicates in the index might cause issues.
    Make sure to remove an annotation from an index beforehand if you need to import it again.
    This can be done with :meth:`~ChildProject.annotations.AnnotationManager.remove_set` to
    remove a set of annotations from the index while preserving raw annotations.

Users are advised to check the consistency and validity of the annotations and their index
using the validation procedure.

Importing any EAF tier
----------------------

When importing EAF annotation files, some tiers are supported by ChildProject, such as `vcm_type` or
`lex_type`. 

If you want to import a tier that is not supported by ChildProject, you can use
:meth:`~ChildProject.annotations.AnnotationManager.import_annotations` as follows :

..code-block:: python

    >>> am.import_annotations(input, new_tiers = ['name_of_tier'])

If a controlled vocabulary is added in the EAF annotation file for this new tier, the values
of the annotations are checked. If a value is not in the controlled vocabulary, it is not
written in the annotation file, and a warning is thrown.
Moreover, the ``metadata/controlled_vocabulary.csv`` dataframe in metadata is either created
with the available controlled vocabularies or updated with this new tier.

If no controlled vocabulary is added in the EAF annoation file, the values are not checked.

Validating annotations
~~~~~~~~~~~~~~~~~~~~~~

The contents of annotations can be searched for errors
using the :meth:`~ChildProject.annotations.AnnotationManager.validate` function.


..code-block:: python

    >>> errors, warnings = am.validate()
    validating BN32_010007_0_0.csv...
    validating BN32_010007_0_0.csv...
    validating BN32_010007_0_0.csv...
    validating BN32_010007_0_3600000.csv...
    validating BN32_010007_0_3600000.csv...
    >>> errors
    []
    >>> warnings
    []

``errors`` and ``warnings`` are empty, indicating that there are no errors.

To gather the errors and warnings raised why validating the index of annotations,
use :meth:`~ChildProject.annotations.AnnotationManager.read`:

..code-block:: python

    >>> errors, warnings = am.read()
    >>> errors
    []
    >>> warnings
    []

Time-of-the-day
~~~~~~~~~~~~~~~

For a number of purposes, it may be convenient to retrieve the timestamp of each vocalization, or to filter out annotations outside
some specific time-range.

Both tasks can be performed through the python API of the package.

Annotations within a specific time-range
----------------------------------------

A given set of annotations may be clipped within a given time-range using :meth:`~ChildProject.annotations.AnnotationManager.get_within_time_range`.
For instance, annotations of audio between 9am and 12am may be retrieved from the following code:

.. code-block:: python

    >>> morning = am.get_within_time_range(am.annotations, '09:00', '12:00')
    >>> morning
            set recording_filename  time_seek  range_onset  range_offset             raw_filename  ...          imported_at  error package_version          start_time  range_onset_time range_offset_time
    0          its    BN32_010007.mp3          0    7320000.0    18120000.0          BN32_010007.its  ...  2021-03-06 22:55:06    NaN           0.0.1 1900-01-01 06:58:00             09:00             12:00
    1          vtc    BN32_010007.mp3          0    7320000.0    18120000.0         BN32_010007.rttm  ...  2021-05-12 19:28:25    NaN           0.0.1 1900-01-01 06:58:00             09:00             12:00
    2          cha    BN32_010007.mp3          0    7320000.0    18120000.0          BN32_010007.cha  ...  2021-05-12 19:39:05    NaN           0.0.1 1900-01-01 06:58:00             09:00             12:00
    3          eaf    BN32_010007.mp3          0   13199449.0    13256801.0          BN32_010007.eaf  ...  2021-07-14 17:39:50    NaN           0.0.1 1900-01-01 06:58:00             10:37      10:38:56.352
    4  cha/aligned    BN32_010007.mp3          0    7320000.0    18120000.0  BN32_010007-aligned.csv  ...  2021-07-15 16:15:48    NaN           0.0.1 1900-01-01 06:58:00             09:00             12:00

    [5 rows x 15 columns]

The onset and offset timestamps for each segments can be calculated with :meth:`~ChildProject.annotations.AnnotationManager.get_segments_timestamps`:

.. code-block:: python

    >>> segments = am.get_segments(morning)
    >>> segments = am.get_segments_timestamps(segments)
    >>> segments[['speaker_type', 'onset_time', 'offset_time']]
        speaker_type              onset_time             offset_time
    0              CHI 2010-07-24 09:00:00.000 2010-07-24 09:20:39.793
    1              CHI 2010-07-24 09:20:39.793 2010-07-24 09:21:43.496
    2              CHI 2010-07-24 09:21:43.496 2010-07-24 09:23:45.168
    3              CHI 2010-07-24 09:23:45.168 2010-07-24 09:24:12.371
    4              CHI 2010-07-24 09:24:12.371 2010-07-24 09:27:27.019
    ...            ...                     ...                     ...
    11801          CHI 2010-07-24 11:56:50.584 2010-07-24 11:56:51.011
    11802          FEM 2010-07-24 11:57:15.749 2010-07-24 11:57:15.992
    11803          MAL 2010-07-24 11:57:24.637 2010-07-24 11:57:25.010
    11804       SPEECH 2010-07-24 11:57:35.237 2010-07-24 11:57:35.666
    11805          CHI 2010-07-24 11:57:35.314 2010-07-24 11:57:35.511

    [11806 rows x 3 columns]


Module reference
~~~~~~~~~~~~~~~~

.. automodule:: ChildProject.annotations
   :members:
   :noindex: