import pandas as pd
import numpy as np

def segments_to_annotation(segments: pd.DataFrame, column: str):
    """Transform a dataframe of annotation segments into a pyannote.core.Annotation object

    :param segments: a dataframe of input segments. It should at least have the following columns: ``segment_onset``, ``segment_offset`` and ``column``.
    :type segments: pd.DataFrame
    :param column: the name of the column in ``segments`` that should be used for the values of the annotations (e.g. speaker_type).
    :type column: str
    :return: the pyannote.core.Annotation object.
    :rtype: pyannote.core.Annotation
    """

    from pyannote.core import Annotation, Segment
    annotation = Annotation()
    
    for segment in segments.to_dict(orient = 'records'):
        start = segment['segment_onset']
        end = segment['segment_offset']
        
        annotation[Segment(start, end)] = segment[column]

    return annotation

def pyannote_metric(segments: pd.DataFrame, reference: str, hypothesis: str, metric, column: str):
    ref = segments_to_annotation(segments[segments['set'] == reference], column)
    hyp = segments_to_annotation(segments[segments['set'] == hypothesis], column)

    return metric(ref, hyp, detailed = True)

def segments_to_grid(
    segments: pd.DataFrame,
    range_onset: int,
    range_offset: int,
    timescale: int,
    column: str,
    categories: list) -> float:

    """Transform a dataframe of annotation segments into a 2d matrix
    representing the indicator function of each of the ``categories`` across
    time.

    Each row of the matrix corresponds to a unit of time of length ``timescale``
    (in milliseconds), ranging from ``range_onset`` to ``range_offset``;
    each column corresponds to one of the ``categories`` provided,
    plus two special columns (overlap and none).

    The value of the cell ``ij`` of the output matrix is set to 1
    if the class ``j`` is active at time ``i``, 0 otherwise.

    The penultimate column (overlap) is set to 1 if more than two
    classes are active at time ``i``.
    The last column (none) is set to one if none of the classes
    are active at time ``i``.

    The shape of the output matrix is therefore
    ``((range_offset-range_onset)/timescale, len(categories) + 2)``.

    The fraction of time a class ``j`` is active can therefore be
    calculated as ``np.mean(grid, axis = 0)[j]``


    :param segments: a dataframe of input segments. It should at least have the following columns: ``segment_onset``, ``segment_offset`` and ``column``.
    :type segments: pd.DataFrame
    :param range_onset: timestamp of the beginning of the range to consider (in milliseconds)
    :type range_onset: int
    :param range_offset: timestamp of the end of the range to consider (in milliseconds)
    :type range_offset: int
    :param timescale: length of each time unit (in milliseconds)
    :type timescale: int
    :param column: the name of the column in ``segments`` that should be used for the values of the annotations (e.g. speaker_type).
    :type column: str
    :param categories: the list of categories
    :type categories: list
    :return: the output grid
    :rtype: numpy.array
    """

    units = int(np.ceil((range_offset-range_onset)/timescale))

    # align on the grid
    segments.loc[:,'segment_onset'] = segments.loc[:,'segment_onset'] - range_onset
    segments.loc[:,'segment_offset'] = segments.loc[:,'segment_offset'] - range_onset

    segments.loc[:,'onset_index'] = (segments.loc[:,'segment_onset'] // timescale).astype(int)
    segments.loc[:,'offset_index'] = (segments.loc[:,'segment_offset'] // timescale).astype(int)

    categories = categories.copy()
    categories.extend(['overlap', 'none'])
    category_table = {
        categories[i]: i
        for i in range(len(categories))
    }

    data = np.zeros((units, len(categories)), dtype = int)
    for segment in segments.to_dict(orient = 'records'):
        category = segment[column]
        if category not in category_table:
            continue

        category_index = category_table[category]
        data[segment['onset_index']:segment['offset_index'], category_index] = 1

    data[:,-2] = np.count_nonzero(data[:,:-2], axis = 1)
    data[:,-1] = data[:,-2] == 0
    data[:,-2] = data[:,-2] > 1

    return data

def grid_to_vector(grid, categories):
    """Transform a grid of active classes into a vector of labels.
    In case several classes are active at time i, the label is 
    set to 'overlap'.

    See :func:`ChildProject.metrics.segments_to_grid` for a description of grids.

    :param grid: a NumPy array of shape ``(n, len(categories))``
    :type grid: numpy.array
    :param categories: the list of categories
    :type categories: list
    :return: the vector of labels of length ``n`` (e.g. ``np.array([none FEM FEM FEM overlap overlap CHI])``)
    :rtype: numpy.array
    """
    return np.vectorize(lambda x: categories[x])(grid.shape[1] - np.argmax(grid[:,::-1], axis = 1) - 1)

def conf_matrix(horizontal_grid, vertical_grid, categories):
    """compute the confusion matrix (as counts) from grids of active classes.

    See :func:`ChildProject.metrics.segments_to_grid` for a description of grids.

    :param horizontal_grid: the grid corresponding to the horizontal axis of the confusion matrix.
    :type horizontal_grid: numpy.array
    :param vertical_grid: the grid corresponding to the vertical axis of the confusion matrix.
    :type vertical_grid: numpy.array
    :param categories: the labels corresponding to each class
    :type categories: list of strings
    :return: a square numpy array of counts
    :rtype: numpy.array
    """
    from sklearn.metrics import confusion_matrix

    vertical = grid_to_vector(vertical_grid, categories)
    horizontal = grid_to_vector(horizontal_grid, categories)

    return confusion_matrix(vertical, horizontal, labels = categories)


def vectors_to_annotation_task(*args):
    """transform vectors of labels into a nltk AnnotationTask object.

    :return: the AnnotationTask object
    :rtype: nltk.metrics.agreement.AnnotationTask
    """
    from nltk.metrics import agreement

    v = np.vstack(args)
    it = np.nditer(v, flags = ['multi_index'])
    data = [
        (it.multi_index[0], it.multi_index[1], str(x))
        for x in it
    ]

    return agreement.AnnotationTask(data = data)

def gamma(segments: pd.DataFrame, column: str, alpha: float = 1, beta: float = 1, precision_level: float = 0.05) -> float:
    """Compute Mathet et al. gamma agreement on `segments`. 
    
    The gamma measure evaluates the reliability of both the segmentation
    and the categorization simultaneously; a extensive description
    of the method and its parameters can be found in Mathet et al., 2015
    (`doi:10.1162/COLI_a_00227 <https://dx.doi.org/10.1162/COLI_a_00227>`_)

    This function uses the `pyagreement-agreement package <https://pygamma-agreement.readthedocs.io/en/latest/>`_
    by `Titeux et al <https://hal.archives-ouvertes.fr/hal-03144116>`_.

    :param segments: input segments dataframe (see :ref:`format-annotations-segments` for the dataframe format)
    :type segments: pd.DataFrame
    :param column: name of the categorical column of the segments to consider, e.g. 'speaker_type'
    :type column: str
    :param alpha: gamma agreement time alignment weight, defaults to 1
    :type alpha: float, optional
    :param beta: gamma agreement categorical weight, defaults to 1
    :type beta: float, optional
    :param precision_level: level of precision (see pygamma-agreement's documentation), defaults to 0.05
    :type precision_level: float, optional
    :return: gamma agreement
    :rtype: float
    """

    from pyannote.core import Segment
    from pygamma_agreement.continuum import Continuum
    from pygamma_agreement.dissimilarity import CombinedCategoricalDissimilarity

    continuum = Continuum()

    for segment in segments.to_dict(orient = 'records'):
        continuum.add(segment['set'], Segment(segment['segment_onset'], segment['segment_offset']), segment[column])

    dissim = CombinedCategoricalDissimilarity(list(continuum.categories),
                                            delta_empty=1,
                                            alpha = alpha,
                                            beta = beta)

    gamma_results = continuum.compute_gamma(dissim, precision_level = precision_level)

    return gamma_results.gamma

