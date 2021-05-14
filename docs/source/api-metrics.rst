Metrics
=======

ChildProject implements several metrics for evaluating annotations and their reliability.
This section demonstrates how to use the python API for these purposes.

.. note:: 

    In order to reproduce the following examples, you will need to install
    the public VanDam corpus using datalad and its annotations:
    
    .. code-block:: bash
    
        datalad install git@gin.g-node.org:/LAAC-LSCP/vandam-data.git
        datalad get vandam-data/annotations


Comparing two annotators
~~~~~~~~~~~~~~~~~~~~~~~~

The performance of automated annotations is usually assessed by comparing them to a ground truth
provided by experts. The ChildProject package provides several tools for such comparisons.

Confusion matrix
----------------

Confusion matrices are widely used to assess the performance of classification algorithms;
they give an accurate visual description of the behavior of a classifier, preserving
most relevant information while still being easy to interpret.

We show how to compute confusion matrices with the ChildProject package,
using data from the VanDam public corpus. In this example, we will compare
annotations from the LENA and the Voice Type Classifier.

The first step is to get all annotations common to the LENA and the VTC.
This can be done with the :meth:`~ChildProject.annotations.AnnotationManager.intersection`
static method 
of :class:`~ChildProject.annotations.AnnotationManager`:

.. code-block:: python

    >>> from ChildProject.projects import ChildProject
    >>> from ChildProject.annotations import AnnotationManager
    >>> from ChildProject.metrics import segments_to_grid, conf_matrix
    >>> 
    >>> speakers = ['CHI', 'OCH', 'FEM', 'MAL']
    >>> 
    >>> project = ChildProject('vandam-data')
    >>> am = AnnotationManager(project)
    >>> am.read()
    ([], ["vandam-data/metadata/annotations.csv: 'chat' is not a permitted value for column 'format' on line 4, should be any of [TextGrid,eaf,vtc_rttm,vcm_rttm,alice,its]", "vandam-data/metadata/annotations.csv: 'custom_rttm' is not a permitted value for column 'format' on line 6, should be any of [TextGrid,eaf,vtc_rttm,vcm_rttm,alice,its]"])
    >>> 
    >>> intersection = AnnotationManager.intersection(am.annotations, ['vtc', 'its'])
    >>> intersection
    set recording_filename  time_seek  range_onset  range_offset      raw_filename    format  filter  annotation_filename          imported_at  error package_version
    2  its    BN32_010007.mp3          0            0      50464512   BN32_010007.its       its     NaN  BN32_010007_0_0.csv  2021-03-06 22:55:06    NaN           0.0.1
    3  vtc    BN32_010007.mp3          0            0      50464512  BN32_010007.rttm  vtc_rttm     NaN  BN32_010007_0_0.csv  2021-05-12 19:28:25    NaN           0.0.1

The next step is to retrieve the contents of the annotations that correspond to the intersection
of the two sets. This is done with :meth:`~ChildProject.annotations.AnnotationManager.get_collapsed_segments`.
This method from :class:`~ChildProject.annotations.AnnotationManager` does the following:

1. Read the contents of all annotations provided into one pandas dataframe.
2. Align them annotator by annotator, allowing cross-comparisons or combination
3. In case these annotations come from non-consecutive portions of audio, or from distinct audio files, they are aligned end-to-end into one virtual timeline.

In the case of the VanDam corpus, there is only one audio file, and it has been entirely annotated by all annotators.
But the following will work even for sparse annotations covering several recordings.

.. code-block:: python

    >>> segments = am.get_collapsed_segments(intersection)
    >>> segments = segments[segments['speaker_type'].isin(speakers)]
    >>> segments
        segment_onset  segment_offset  speaker_id  ling_type speaker_type  vcm_type  lex_type  ...          imported_at  error  package_version  abs_range_onset  abs_range_offset    duration  position
    1             9730.0         10540.0         NaN        NaN          OCH       NaN       NaN  ...  2021-03-06 22:55:06    NaN            0.0.1                0          50464512  50464512.0       0.0
    15           35820.0         36930.0         NaN        NaN          OCH       NaN       NaN  ...  2021-03-06 22:55:06    NaN            0.0.1                0          50464512  50464512.0       0.0
    21           67020.0         67620.0         NaN        NaN          OCH       NaN       NaN  ...  2021-03-06 22:55:06    NaN            0.0.1                0          50464512  50464512.0       0.0
    25           71640.0         72240.0         NaN        NaN          FEM       NaN       NaN  ...  2021-03-06 22:55:06    NaN            0.0.1                0          50464512  50464512.0       0.0
    29           87370.0         88170.0         NaN        NaN          OCH       NaN       NaN  ...  2021-03-06 22:55:06    NaN            0.0.1                0          50464512  50464512.0       0.0
    ...              ...             ...         ...        ...          ...       ...       ...  ...                  ...    ...              ...              ...               ...         ...       ...
    22342     50122992.0      50123518.0         NaN        NaN          FEM       NaN       NaN  ...  2021-05-12 19:28:25    NaN            0.0.1                0          50464512  50464512.0       0.0
    22344     50152103.0      50153510.0         NaN        NaN          FEM       NaN       NaN  ...  2021-05-12 19:28:25    NaN            0.0.1                0          50464512  50464512.0       0.0
    22348     50233080.0      50234492.0         NaN        NaN          FEM       NaN       NaN  ...  2021-05-12 19:28:25    NaN            0.0.1                0          50464512  50464512.0       0.0
    22350     50325867.0      50325989.0         NaN        NaN          CHI       NaN       NaN  ...  2021-05-12 19:28:25    NaN            0.0.1                0          50464512  50464512.0       0.0
    22352     50356380.0      50357011.0         NaN        NaN          FEM       NaN       NaN  ...  2021-05-12 19:28:25    NaN            0.0.1                0          50464512  50464512.0       0.0

    [20887 rows x 44 columns]


For an efficient computation of the confusion matrix, the timeline is then split into chunks of a given length
(in our case, we will set the time steps to 100 milliseconds).
This is done with :func:`ChildProject.metrics.segments_to_grid`, which transforms a dataframe of segments
into a matrix of the indicator functions of each classification category at each time unit. This function adds to more
categories: 'overlap' and 'none'.

.. code-block:: python

    >>> vtc = segments_to_grid(segments[segments['set'] == 'vtc'], 0, segments['segment_offset'].max(), 100, 'speaker_type', speakers)
    /Users/acristia/anaconda3/lib/python3.7/site-packages/pandas/core/indexing.py:1676: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    self._setitem_single_column(ilocs[0], value, pi)
    /Users/acristia/anaconda3/lib/python3.7/site-packages/pandas/core/indexing.py:1597: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    self.obj[key] = value
    >>> its = segments_to_grid(segments[segments['set'] == 'its'], 0, segments['segment_offset'].max(), 100, 'speaker_type', speakers)
    >>> 
    >>> vtc.shape
    (503571, 6)
    >>> vtc
    array([[0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
        ...,
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1]])
    >>> 


We can now compute the confusion matrix:

.. code-block:: python

>>> speakers.extend(['overlap', 'none'])
>>> confusion_counts = conf_matrix(its, vtc, speakers)
>>> confusion_counts
array([[ 17802,   5139,   2537,    566,      0,  17392],
       [   178,   1329,    129,     65,      0,   1947],
       [   998,    818,  14530,   1964,      0,  16010],
       [   158,    155,   1984,  14613,      0,  10918],
       [  2852,   2407,   4390,   3203,      0,   5138],
       [  3053,   2158,   3674,   2464,      0, 365000]])
>>> 

Using pyannote.metrics
----------------------

Many metrics are then used to measure the performance (recall, precision, accuracy, etc.).

Reliability evaluations
~~~~~~~~~~~~~~~~~~~~~~~



Module reference
~~~~~~~~~~~~~~~~

.. automodule:: ChildProject.metrics
   :members:
   :noindex:
