import pandas as pd

def gamma(segments: pd.DataFrame, column: str, alpha = 1, beta = 1, precision_level = 0.05) -> float:
    """compute gamma agreement on `segments`. (10.1162/COLI_a_00227,https://hal.archives-ouvertes.fr/hal-03144116) 

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

def segments_to_annotation(segments: pd.DataFrame, column: str) -> Annotation:
    from pyannote.core import Annotation, Segment

    annotation = Annotation()
    
    for segment in segments.to_dict(orient = 'records'):
        start = segment['segment_onset']
        end = segment['segment_offset']

        if 'time_seek' in segment:
            start += segment['time_seek']
            end += segment['time_seek']
        
        if 'session_offset' in segment:
            start += segment['session_offset']
            end += segment['session_offset']
        
        annotation[Segment(start, end)] = segment[column]

    return annotation

def pyannote_metric(segments: pd.DataFrame, reference: str, hypothesis: str, metric, column: str):
    ref = segments_to_annotation(segments[segments['set'] == reference], column)
    hyp = segments_to_annotation(segments[segments['set'] == hypothesis], column)

    return metric(ref, hyp, detailed = True)