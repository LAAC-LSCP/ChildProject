Metrics
=======

ChildProject implements several metrics for evaluating annotations and their reliability.
This section demonstrates how to use the python API for these purposes.

.. note:: 

    In order to reproduce the following examples, you will need to install
    the public VanDam corpus and its annotations using datalad:
    
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
    >>> speakers = ['CHI', 'OCH', 'FEM', 'MAL']
    >>> project = ChildProject('vandam-data')
    >>> am = AnnotationManager(project)
    >>> am.read()
    ([], ["vandam-data/metadata/annotations.csv: 'chat' is not a permitted value for column 'format' on line 4, should be any of [TextGrid,eaf,vtc_rttm,vcm_rttm,alice,its]", "vandam-data/metadata/annotations.csv: 'custom_rttm' is not a permitted value for column 'format' on line 6, should be any of [TextGrid,eaf,vtc_rttm,vcm_rttm,alice,its]"])
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
into a matrix of the indicator functions of each classification category at each time unit.

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
    >>> vtc.shape
    (503571, 5)
    >>> vtc
    array([[0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        ...,
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1]])

Note that this matrix has 5 columns, although there are only 4 categories (CHI, OCH, FEM and MAL).
This is because :func:`~ChildProject.metrics.segments_to_grid` appends the matrix with a 'none' none,
which is set to 1 when all classes are inactive.
It can be turned off by setting `none = False`. It is possible to add an 'overlap' column
by setting `overlap=True`; this column is set to 1 when at least 2 classes are active.

We can now compute the confusion matrix:

.. code-block:: python

    >>> confusion_counts = conf_matrix(vtc, its)
    >>> confusion_counts
    array([[ 20503,   7285,   4296,   1191,  21062],
    [  1435,   3354,    704,    136,   4105],
    [  2700,   1414,  18442,   4649,  19080],
    [   323,    229,   4600,  17654,  12415],
    [  3053,   2158,   3674,   2464, 365000]])

This means that 20503 of the 100 ms chunks were labelled as containing CHI speech
by both the VTC and the LENA; 7285 chunks have been labelled as containing CHI speech by the VTC
while being labelled as OCH by the LENA.

It is sometimes more useful to normalize confusion matrices:

    >>> import numpy as np
    >>> normalized = confusion_counts/(np.sum(vtc, axis = 0)[:,None])
    >>> rel
    array([[0.37733036, 0.13407071, 0.07906215, 0.02191877, 0.38761801],
    [0.14742141, 0.34456544, 0.07232381, 0.01397165, 0.42171769],
    [0.05833423, 0.03054985, 0.39844442, 0.10044291, 0.41222858],
    [0.00917067, 0.0065018 , 0.1306039 , 0.50123506, 0.35248857],
    [0.00811215, 0.00573404, 0.00976222, 0.00654711, 0.96984448]])

The top-left cell now reads as: 37,8% of the 100 ms chunks labelled as CHI by the VTC
are also labelled as CHI by the LENA.

Summing 

Using pyannote.metrics
----------------------

Confusion matrices are still dimensional data
(with :math:`n \times n` components for :math:`n` labels),
which renders performance comparisons of several annotators
difficult: it is hard to tell which one of two classifiers
is the closest to the ground truth using confusion matrices.

As a result, in Machine Learning, many scalar measures are used
in order to assess the overall performance of a classifier.
These include recall, precision, accuracy, etc.

The `pyannote-metrics package <https://pyannote.github.io/pyannote-metrics/>`_
implements many of the metrics that are typically used in speech processing.
ChildProject interfaces well with pyannote-metrics. Below, we show how 
to use both packages to compute recall and precision.

The first step is to convert the dataframe of segments into one :meth:`pyannote.core.Annotation`
object per annotator:

.. code-block:: python

    >>> from ChildProject.metrics import segments_to_annotation
    >>> ref = segments_to_annotation(segments[segments['set'] == 'vtc'], 'speaker_type')
    >>> hyp = segments_to_annotation(segments[segments['set'] == 'its'], 'speaker_type')

Now, any pyannote metric can be instantianted and used with these annotations:

.. code-block:: python

    >>> from pyannote.metrics.detection import DetectionPrecisionRecallFMeasure
    >>> metric = DetectionPrecisionRecallFMeasure()
    >>> detail = metric.compute_components(ref, hyp)
    >>> precision, recall, f = metric.compute_metrics(detail)
    >>> print(f'{precision:.2f}/{recall:.2f}/{f:.2f}')
    0.87/0.60/0.71



Reliability evaluations
~~~~~~~~~~~~~~~~~~~~~~~



Module reference
~~~~~~~~~~~~~~~~

.. automodule:: ChildProject.metrics
   :members:
   :noindex:
