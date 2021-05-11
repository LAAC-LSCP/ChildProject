import pandas as pd
import numpy as np

from .annotations import AnnotationManager

#from nltk.metrics.agreement import AnnotationTask

def gamma(segments: pd.DataFrame, column: str, alpha = 1, beta = 1, precision_level = 0.05) -> float:
    """compute gamma agreement on `segments`. (doi:10.1162/COLI_a_00227,https://hal.archives-ouvertes.fr/hal-03144116) 

    :param segments: input segments dataframe (see :ref:`format-annotations-segments` for the dataframe format)
    :type segments: pd.DataFrame
    :param column: name of the categorical column of the segments to consider, e.g. 'speaker_type'
    :type column: str
    :param alpha: gamma agreement time alignment weight, defaults to 1
    :type alpha: int, optional
    :param beta: gamma agreement categorical weight, defaults to 1
    :type beta: int, optional
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

def segments_to_annotation(segments: pd.DataFrame, column: str):
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