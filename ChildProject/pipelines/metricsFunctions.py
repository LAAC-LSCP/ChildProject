import pandas as pd
import numpy as np
import ast
import re
import functools
from typing import Union, Set, Tuple

"""
This file lists all the metrics functions commonly used.
New metrics can be added by defining new functions for the Metrics class to use :
 - Create a new function using the same arguments (i.e. annotations, duration, **kwargs)
 - Define calculation of the metric with:
     - annotations, which is a dataframe containing all the relevant annotated segments  to use. It contains the
       annotation content (https://childproject.readthedocs.io/en/latest/format.html#id10) joined with the annotation
       index info (https://childproject.readthedocs.io/en/latest/format.html#id11) as well as any column that was
       requested to be added to the results by the user using --child-cols or --rec-cols (eg --child-cols child_dob,
       languages will make columns 'child_dob' and 'languages' available)
     - duration which is the duration of audio annotated in milliseconds
     - kwargs, whatever keyword parameter you chose to pass to the function (except 'name', 'callable', 'set' which can 
       not be used). This will need to be given with the list of metrics when called
 - Wrap you function with the 'metricFunction' decorator to make it callable by the pipeline, read metricFunction help 
   for more info
   
!! Metrics functions should still behave and return the correct result when receiving an empty dataframe
"""

# error message in case of missing columns in annotations
MISSING_COLUMNS = 'The given set <{}> does not have the required column(s) <{}> for computing the {} metric'

RESERVED = {'set', 'name', 'callable'}  # arguments reserved usage. use other keyword labels.


def metricFunction(args: set, columns: Union[Set[str], Tuple[Set[str], ...]], empty_value=0, default_name: str = None):
    """Decorator for all metrics functions to make them ready to be called by the pipeline.
    
    :param args: set of required keyword arguments for that function, raise ValueError if were not given \
    you cannot use keywords [name, callable, set] as they are reserved
    :type args: set
    :param columns: required columns in the dataframe given, missing columns raise ValueError
    :type columns: set
    :param default_name: default name to use for the metric in the resulting dataframe. Every keyword argument found in the name will be replaced by its value (e.g. 'voc_speaker_ph' uses kwarg 'speaker' so if speaker = 'CHI', name will be 'voc_chi_ph'). if no name is given, the __name__ of the function is used
    :type default_name: str
    :param empty_value: value to return when annotations are empty but the unit was annotated (e.g. 0 for counts like voc_speaker_ph , None for proportions like lp_n)
    :type empty_value: float|int
    :return: new function to substitute the metric function
    :rtype: Callable
    """

    def decorator(function):
        for a in args:
            if a in RESERVED:
                raise ValueError(
                    'Error when defining {} with required argument {}, you cannot use reserved keywords {},\
                     change your required argument name'.format(
                        function.__name__, a, RESERVED))

        @functools.wraps(function)
        def new_func(annotations: pd.DataFrame, duration: int, **kwargs):
            for arg in args:
                if arg not in kwargs:
                    raise ValueError(f"{function.__name__} metric needs an argument <{arg}>")
            # if a name is explicitly given, use it
            if 'name' in kwargs and not pd.isnull(kwargs['name']) and kwargs['name']:
                metric_name = kwargs['name']
            # else if a default name for the function exists, use the function name
            elif default_name:
                metric_name = default_name
            # else, no name was found, use the name of the function
            else:
                metric_name = function.__name__

            metric_name_replaced = metric_name
            # metric_name is the basename used to designate this metric (voc_speaker_ph),
            # metric_name_replaced replaces the values of kwargs
            # found in the name by their values, giving the metric name for that instance only (voc_chi_ph)
            for arg in kwargs:
                metric_name_replaced = re.sub(arg, str(kwargs[arg]).lower(), metric_name_replaced)
            if annotations.shape[0]:
                # if multiple possibilities of columns, explore each and fail only if each combination is missing
                # a column, if one possibility, fail if a column is missing
                if isinstance(columns, tuple) and len(columns) > 0 and isinstance(columns[0], set):
                    missing_columns = []
                    for possible_cols in columns:
                        possible_missing = possible_cols - set(annotations.columns)
                        if possible_missing:
                            missing_columns.append(possible_missing)
                    # if we have as many cases of missing columns as possibilities, we can't compute the metric
                    if len(missing_columns) == len(columns):
                        raise ValueError(
                            MISSING_COLUMNS.format(annotations['set'].iloc[0],
                                                   ' or '.join([str(s) for s in missing_columns]),
                                                   metric_name))
                else:
                    missing_columns = columns - set(annotations.columns)
                    if missing_columns:
                        raise ValueError(MISSING_COLUMNS.format(annotations['set'].iloc[0], missing_columns, metric_name))
                res = function(annotations, duration, **kwargs)
            else:  # no annotation for that unit
                res = empty_value if duration else None  # duration != 0 => was annotated but not segments there
            return metric_name_replaced, res

        return new_func

    return decorator


def peak_hour_metric(empty_value=0):
    """
    empty_value : should repeat the empty value of the metric function wrapper (as this will be used for empty periods)
    """
    def decorator(function):
        """Decorator a metric function to select the maximum value observed over 1h periods. function is prefixed with
           'peak_'
            """
        @functools.wraps(function)
        def new_func(annotations: pd.DataFrame, duration: int, **kwargs):
            # time to consider for periods, here 1h by default, else put it in kwargs
            period_time = 3600000 if 'period_time' not in kwargs else kwargs['period_time']
            periods = duration // period_time  # number of hours to consider

            # what hour it belongs to (we made the choice of using onset to choose the hour)
            annotations['hour_number_metric'] = annotations['segment_onset'] // period_time

            result_array = np.array([])
            for i in range(periods):
                # select the annotations for this hour
                period_annotations = annotations[annotations['hour_number_metric'] == i]

                if period_annotations.shape[0]:
                    # compute metric for the period
                    metric = function(period_annotations, period_time, **kwargs)
                else:
                    metric = empty_value

                result_array = np.append(result_array, metric)  # store the result

            # if we have results, return the max, else return NaN
            if len(result_array):
                return np.nanmax(result_array)
            else:
                return np.nan

        # wraps will give the same name and doc, so we need to slightly edit them for the peak function
        new_func.__doc__ = "Computing the peak for 1h for the following metric:\n\n" + function.__doc__
        new_func.__name__ = "peak_" + function.__name__
        new_func.__qualname__ = "peak_" + function.__qualname__
        return new_func
    return decorator


def per_hour_metric():
    """
    """
    def decorator(function):
        """Decorator creating a metric function controlling the original value by time. function is suffixed with '_ph'
            """
        @functools.wraps(function)
        def new_func(annotations: pd.DataFrame, duration: int, **kwargs):
            # time to consider for periods, here 1h by default, else put it in kwargs
            return function(annotations, duration, **kwargs) * (3600000 / duration)

        # wraps will give the same name and doc, so we need to slightly edit them for the peak function
        new_func.__doc__ = function.__doc__ + "\nThis value is a 'per hour' value."
        new_func.__name__ = function.__name__ + '_ph'
        new_func.__qualname__ = function.__qualname__ + '_ph'
        return new_func
    return decorator


def voc_speaker(annotations: pd.DataFrame, duration: int, **kwargs):
    """number of vocalizations for a given speaker type
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    return annotations[annotations["speaker_type"] == kwargs["speaker"]].shape[0]


# Decorate for the peak metric, per hour metric, and then the classic metric to avoid conflicts of decoration
peak_voc_speaker = metricFunction({"speaker"}, {"speaker_type"})(peak_hour_metric()(voc_speaker))
voc_speaker_ph = metricFunction({"speaker"}, {"speaker_type"})(per_hour_metric()(voc_speaker))
voc_speaker = metricFunction({"speaker"}, {"speaker_type"})(voc_speaker)


def voc_dur_speaker(annotations: pd.DataFrame, duration: int, **kwargs):
    """total duration of vocalizations by a given speaker type in milliseconds per hour
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    return annotations[annotations["speaker_type"] == kwargs["speaker"]]["duration"].sum()


# Decorate for the peak metric, per hour metric, and then the classic metric to avoid conflicts of decoration
peak_voc_dur_speaker = metricFunction({"speaker"}, {"speaker_type", "duration"})(peak_hour_metric()(voc_dur_speaker))
voc_dur_speaker_ph = metricFunction({"speaker"}, {"speaker_type", "duration"})(per_hour_metric()(voc_dur_speaker))
voc_dur_speaker = metricFunction({"speaker"}, {"speaker_type", "duration"})(voc_dur_speaker)


@metricFunction({"speaker"}, {"speaker_type", "duration"}, np.nan)
def avg_voc_dur_speaker(annotations: pd.DataFrame, duration: int, **kwargs):
    """average duration in milliseconds of vocalizations for a given speaker type 
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    return annotations[annotations["speaker_type"] == kwargs["speaker"]]["duration"].mean()


def wc_speaker(annotations: pd.DataFrame, duration: int, **kwargs):
    """number of words for a given speaker type
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    return annotations[annotations["speaker_type"] == kwargs["speaker"]]["words"].sum()


peak_wc_speaker = metricFunction({"speaker"}, {"speaker_type", "words"})(peak_hour_metric()(wc_speaker))
wc_speaker_ph = metricFunction({"speaker"}, {"speaker_type", "words"})(per_hour_metric()(wc_speaker))
wc_speaker = metricFunction({"speaker"}, {"speaker_type", "words"})(wc_speaker)


def sc_speaker(annotations: pd.DataFrame, duration: int, **kwargs):
    """number of syllables for a given speaker type
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    return annotations[annotations["speaker_type"] == kwargs["speaker"]]["syllables"].sum()


peak_sc_speaker = metricFunction({"speaker"}, {"speaker_type", "syllables"})(peak_hour_metric()(sc_speaker))
sc_speaker_ph = metricFunction({"speaker"}, {"speaker_type", "syllables"})(per_hour_metric()(sc_speaker))
sc_speaker = metricFunction({"speaker"}, {"speaker_type", "syllables"})(sc_speaker)


def pc_speaker(annotations: pd.DataFrame, duration: int, **kwargs):
    """number of phonemes for a given speaker type
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    return annotations[annotations["speaker_type"] == kwargs["speaker"]]["phonemes"].sum()


peak_pc_speaker = metricFunction({"speaker"}, {"speaker_type", "phonemes"})(peak_hour_metric()(pc_speaker))
pc_speaker_ph = metricFunction({"speaker"}, {"speaker_type", "phonemes"})(per_hour_metric()(pc_speaker))
pc_speaker = metricFunction({"speaker"}, {"speaker_type", "phonemes"})(pc_speaker)


def wc_adu(annotations: pd.DataFrame, duration: int, **kwargs):
    """number of words for all speakers
    
    Required keyword arguments:
    """
    return annotations["words"].sum()


peak_wc_adu = metricFunction(set(), {"words"})(peak_hour_metric()(wc_adu))
wc_adu_ph = metricFunction(set(), {"words"})(per_hour_metric()(wc_adu))
wc_adu = metricFunction(set(), {"words"})(wc_adu)


def sc_adu(annotations: pd.DataFrame, duration: int, **kwargs):
    """number of syllables for all speakers
    
    Required keyword arguments:
    """
    return annotations["syllables"].sum()


peak_sc_adu = metricFunction(set(), {"syllables"})(peak_hour_metric()(sc_adu))
sc_adu_ph = metricFunction(set(), {"syllables"})(per_hour_metric()(sc_adu))
sc_adu = metricFunction(set(), {"syllables"})(sc_adu)


def pc_adu(annotations: pd.DataFrame, duration: int, **kwargs):
    """number of phonemes for all speakers
    
    Required keyword arguments:
    """
    return annotations["phonemes"].sum()


peak_pc_adu = metricFunction(set(), {"phonemes"})(peak_hour_metric()(pc_adu))
pc_adu_ph = metricFunction(set(), {"phonemes"})(per_hour_metric()(pc_adu))
pc_adu = metricFunction(set(), {"phonemes"})(pc_adu)


def cry_voc_speaker(annotations: pd.DataFrame, duration: int, **kwargs):
    """number of cry vocalizations for a given speaker (based on vcm_type or lena cries)
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    if 'vcm_type' in annotations.columns:
        return annotations.loc[(annotations["speaker_type"] == kwargs["speaker"]) &
                               (annotations["vcm_type"] == "Y")].shape[0]
    # elif 'cries' in annotations.columns:
    else:
        return annotations[annotations['speaker_type'] == kwargs["speaker"]]["cries"].apply(lambda x: len(ast.literal_eval(x))).sum()


peak_cry_voc_speaker = metricFunction({"speaker"}, ({"speaker_type", "vcm_type"}, {"speaker_type", "cries"})
                                      )(peak_hour_metric()(cry_voc_speaker))
cry_voc_speaker_ph = metricFunction({"speaker"}, ({"speaker_type", "vcm_type"}, {"speaker_type", "cries"})
                                    )(per_hour_metric()(cry_voc_speaker))
cry_voc_speaker = metricFunction({"speaker"}, ({"speaker_type", "vcm_type"}, {"speaker_type", "cries"})
                                 )(cry_voc_speaker)


def cry_voc_dur_speaker(annotations: pd.DataFrame, duration: int, **kwargs):
    """total duration of cry vocalizations by a given speaker type in milliseconds (based on vcm_type or lena cry)
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    if 'vcm_type' in annotations.columns and 'duration' in annotations.columns:
        return annotations.loc[(annotations["speaker_type"] == kwargs["speaker"]) &
                               (annotations["vcm_type"] == "Y")]["duration"].sum()
    # elif 'child_cry_vfx_len' in annotations.columns:
    else:
        return annotations[annotations['speaker_type'] == kwargs["speaker"]]["child_cry_vfx_len"].sum()


peak_cry_voc_dur_speaker = metricFunction({"speaker"}, ({"speaker_type", "vcm_type", "duration"}, {"speaker_type", "child_cry_vfx_len"}))(
    peak_hour_metric()(cry_voc_dur_speaker))
cry_voc_dur_speaker_ph = metricFunction({"speaker"}, ({"speaker_type", "vcm_type", "duration"}, {"speaker_type", "child_cry_vfx_len"}))(
    per_hour_metric()(cry_voc_dur_speaker))
cry_voc_dur_speaker = metricFunction({"speaker"}, ({"speaker_type", "vcm_type", "duration"}, {"speaker_type", "child_cry_vfx_len"}))(cry_voc_dur_speaker)


@metricFunction({"speaker"}, ({"speaker_type", "vcm_type", "duration"}, {'speaker_type', "child_cry_vfx_len", "cries"}), np.nan)
def avg_cry_voc_dur_speaker(annotations: pd.DataFrame, duration: int, **kwargs):
    """average duration of cry vocalizations by a given speaker type (based on vcm_type or lena cries)
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    if 'vcm_type' in annotations.columns and 'duration' in annotations.columns:
        value = annotations.loc[(annotations["speaker_type"] == kwargs["speaker"]) &
                                (annotations["vcm_type"] == "Y")]["duration"].mean()
    else:
        annots = annotations[annotations['speaker_type'] == kwargs["speaker"]]
        value = annots["child_cry_vfx_len"].sum() / annots["cries"].apply(lambda x: len(ast.literal_eval(x))).sum()

    if pd.isnull(value):
        value = 0
    return value


def can_voc_speaker(annotations: pd.DataFrame, duration: int, **kwargs):
    """number of canonical vocalizations for a given speaker type (based on vcm_type)
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    return annotations.loc[(annotations["speaker_type"] == kwargs["speaker"]) & (annotations["vcm_type"] == "C")].shape[
        0]


peak_can_voc_speaker = metricFunction({"speaker"}, {"speaker_type", "vcm_type"})(peak_hour_metric()(can_voc_speaker))
can_voc_speaker_ph = metricFunction({"speaker"}, {"speaker_type", "vcm_type"})(per_hour_metric()(can_voc_speaker))
can_voc_speaker = metricFunction({"speaker"}, {"speaker_type", "vcm_type"})(can_voc_speaker)


def can_voc_dur_speaker(annotations: pd.DataFrame, duration: int, **kwargs):
    """total duration of canonical vocalizations by a given speaker type in milliseconds (based on vcm_type)
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    return annotations.loc[(annotations["speaker_type"] == kwargs["speaker"]) & (annotations["vcm_type"] == "C")][
        "duration"].sum()


peak_can_voc_dur_speaker = metricFunction({"speaker"}, {"speaker_type", "vcm_type", "duration"})(
    peak_hour_metric()(can_voc_dur_speaker))
can_voc_dur_speaker_ph = metricFunction({"speaker"}, {"speaker_type", "vcm_type", "duration"})(
    per_hour_metric()(can_voc_dur_speaker))
can_voc_dur_speaker = metricFunction({"speaker"}, {"speaker_type", "vcm_type", "duration"})(can_voc_dur_speaker)


@metricFunction({"speaker"}, {"speaker_type", "vcm_type", "duration"}, np.nan)
def avg_can_voc_dur_speaker(annotations: pd.DataFrame, duration: int, **kwargs):
    """average duration of canonical vocalizations for a given speaker type (based on vcm_type)
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    value = annotations.loc[(annotations["speaker_type"] == kwargs["speaker"]) & (annotations["vcm_type"] == "C")][
        "duration"].mean()
    if pd.isnull(value): value = 0
    return value


def non_can_voc_speaker(annotations: pd.DataFrame, duration: int, **kwargs):
    """number of non-canonical vocalizations for a given speaker type (based on vcm_type)
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    return annotations.loc[(annotations["speaker_type"] == kwargs["speaker"]) &
                           (annotations["vcm_type"] == "N")].shape[0]


peak_non_can_voc_speaker = metricFunction({"speaker"}, {"speaker_type", "vcm_type"})(
    peak_hour_metric()(non_can_voc_speaker))
non_can_voc_speaker_ph = metricFunction({"speaker"}, {"speaker_type", "vcm_type"})(
    per_hour_metric()(non_can_voc_speaker))
non_can_voc_speaker = metricFunction({"speaker"}, {"speaker_type", "vcm_type"})(non_can_voc_speaker)


def non_can_voc_dur_speaker(annotations: pd.DataFrame, duration: int, **kwargs):
    """total duration of non-canonical vocalizations by a given speaker type in milliseconds (based on vcm_type)
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    return annotations.loc[(annotations["speaker_type"] == kwargs["speaker"]) &
                           (annotations["vcm_type"] == "N")]["duration"].sum()


peak_non_can_voc_dur_speaker = metricFunction({"speaker"}, {"speaker_type", "vcm_type", "duration"})(
    peak_hour_metric()(non_can_voc_dur_speaker))
non_can_voc_dur_speaker_ph = metricFunction({"speaker"}, {"speaker_type", "vcm_type", "duration"})(
    per_hour_metric()(non_can_voc_dur_speaker))
non_can_voc_dur_speaker = metricFunction({"speaker"}, {"speaker_type", "vcm_type", "duration"})(non_can_voc_dur_speaker)


@metricFunction({"speaker"}, {"speaker_type", "vcm_type", "duration"}, np.nan)
def avg_non_can_voc_dur_speaker(annotations: pd.DataFrame, duration: int, **kwargs):
    """average duration of non-canonical vocalizations for a given speaker type (based on vcm_type)
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    value = annotations.loc[(annotations["speaker_type"] == kwargs["speaker"]) &
                            (annotations["vcm_type"] == "N")]["duration"].mean()
    if pd.isnull(value):
        value = 0
    return value


@metricFunction(set(), set(), np.nan)
def lp_n(annotations: pd.DataFrame, duration: int, **kwargs):
    """linguistic proportion on the number of vocalizations for CHI (based on vcm_type or [cries,vfxs,utterances_count] if vcm_type does not exist)
    
    Required keyword arguments:
    """
    if {"cries", "vfxs", "utterances_count"}.issubset(annotations.columns):
        annotations = annotations[annotations["speaker_type"] == "CHI"]
        cries = annotations["cries"].apply(lambda x: len(ast.literal_eval(x))).sum()
        vfxs = annotations["vfxs"].apply(lambda x: len(ast.literal_eval(x))).sum()
        utterances = annotations["utterances_count"].sum()
        total = (utterances + cries + vfxs)
        if total:
            value = utterances / total
        else:
            value = np.nan
    elif "vcm_type" in annotations.columns:
        speech_voc = annotations.loc[(annotations["speaker_type"] == "CHI") &
                                     (annotations["vcm_type"].isin(["N", "C"]))].shape[0]
        cry_voc = annotations.loc[(annotations["speaker_type"] == "CHI") & (annotations["vcm_type"] == "Y")].shape[0]
        total = speech_voc + cry_voc
        if total:
            value = speech_voc / total
        else:
            value = np.nan
    else:
        raise ValueError(
            "the given set does not have the necessary columns for this metric, choose a set that contains either ["
            "vcm_type] or [cries,vfxs,utterances_count]")
    return value


@metricFunction(set(), {"speaker_type", "vcm_type"}, np.nan)
def cp_n(annotations: pd.DataFrame, duration: int, **kwargs):
    """canonical proportion on the number of vocalizations for CHI (based on vcm_type)
    
    Required keyword arguments:
    """
    speech_voc = annotations.loc[(annotations["speaker_type"] == "CHI") &
                                 (annotations["vcm_type"].isin(["N", "C"]))].shape[0]
    can_voc = annotations.loc[(annotations["speaker_type"] == "CHI") & (annotations["vcm_type"] == "C")].shape[0]
    if speech_voc:
        value = can_voc / speech_voc
    else:
        value = np.nan
    return value


@metricFunction(set(), set(), np.nan)
def lp_dur(annotations: pd.DataFrame, duration: int, **kwargs):
    """linguistic proportion on the duration of vocalizations for CHI (based on vcm_type or [child_cry_vfxs_len,utterances_length] if vcm_type does not exist)
    
    Required keyword arguments:
    """
    if {"child_cry_vfx_len", "utterances_length"}.issubset(annotations.columns):
        annotations = annotations[annotations["speaker_type"] == "CHI"]
        utter_len = annotations["utterances_length"].sum()
        total = annotations["child_cry_vfx_len"].sum() + utter_len
        if total:
            value = utter_len / total
        else:
            value = np.nan
    elif "vcm_type" in annotations.columns:
        speech_dur = annotations.loc[(annotations["speaker_type"] == "CHI") &
                                     (annotations["vcm_type"].isin(["N", "C"]))]["duration"].sum()
        cry_dur = annotations.loc[(annotations["speaker_type"] == "CHI") &
                                  (annotations["vcm_type"] == "Y")]["duration"].sum()
        total = speech_dur + cry_dur
        if total:
            value = speech_dur / total
        else:
            value = np.nan
    else:
        raise ValueError(
            "the {} set does not have the necessary columns for this metric, choose a set that contains either ["
            "vcm_type] or [child_cry_vfx_len,utterances_length]")
    return value


@metricFunction(set(), {"speaker_type", "vcm_type", "duration"}, np.nan)
def cp_dur(annotations: pd.DataFrame, duration: int, **kwargs):
    """canonical proportion on the number of vocalizations for CHI (based on vcm_type)
    
    Required keyword arguments:
    """
    speech_dur = annotations.loc[(annotations["speaker_type"] == "CHI") &
                                 (annotations["vcm_type"].isin(["N", "C"]))]["duration"].sum()
    can_dur = annotations.loc[(annotations["speaker_type"] == "CHI") &
                              (annotations["vcm_type"] == "C")]["duration"].sum()
    if speech_dur:
        value = can_dur / speech_dur
    else:
        value = np.nan
    return value


def lena_CVC(annotations: pd.DataFrame, duration: int, **kwargs):
    """number of child vocalizations according to LENA's extraction
    
    Required keyword arguments:
    """
    return annotations["utterances_count"].sum()


peak_lena_CVC = metricFunction(set(), {"utterances_count"})(peak_hour_metric()(lena_CVC))
lena_CVC_ph = metricFunction(set(), {"utterances_count"})(per_hour_metric()(lena_CVC))
lena_CVC = metricFunction(set(), {"utterances_count"})(lena_CVC)


def lena_CTC(annotations: pd.DataFrame, duration: int, **kwargs):
    """number of conversational turn counts according to LENA's extraction
    
    Required keyword arguments:
    """
    conv_types = {'TIMR', 'TIFR'}
    return annotations[annotations["lena_conv_turn_type"].isin(conv_types)].shape[0]


peak_lena_CTC = metricFunction(set(), {"lena_conv_turn_type"})(peak_hour_metric()(lena_CTC))
lena_CTC_ph = metricFunction(set(), {"lena_conv_turn_type"})(per_hour_metric()(lena_CTC))
lena_CTC = metricFunction(set(), {"lena_conv_turn_type"})(lena_CTC)


def simple_CTC(annotations: pd.DataFrame,
        duration: int,
        interlocutors_1=('CHI',),
        interlocutors_2=('FEM', 'MAL', 'OCH'),
        max_interval=1000,
        min_delay=0,
        **kwargs):
    """number of conversational turn counts based on vocalizations occurring
    in a given interval of one another

    keyword arguments:
        - interlocutors_1 : first group of interlocutors, default = ['CHI']
        - interlocutors_2 : second group of interlocutors, default = ['FEM','MAL','OCH']
        - max_interval : maximum interval in ms for it to be considered a turn, default = 1000
        - min_delay : minimum delay between somebody starting speaking
    """
    # build the interactants groups, every label in interlocutors_1 can interact with interlocutors_2 and vice versa
    speakers = set(interlocutors_1 + interlocutors_2)
    interactants = {k: set(interlocutors_2) for k in interlocutors_1}
    for k in interlocutors_2:
        if k in interactants:
            interactants[k] = interactants[k] | set(interlocutors_1)
        else:
            interactants[k] = set(interlocutors_1)

    annotations = annotations[annotations["speaker_type"].isin(speakers)].copy()

    if annotations.shape[0]:
        # store the duration between vocalizations
        annotations["iti"] = annotations["segment_onset"] - annotations["segment_offset"].shift(1)
        # store the previous speaker
        annotations["prev_speaker_type"] = annotations["speaker_type"].shift(1)

        annotations["delay"] = annotations["segment_onset"] - annotations["segment_onset"].shift(1)

        # not using absolute value for 'iti' is a choice and should be evaluated (we allow speakers to 'interrupt'
        # themselves
        annotations["is_CT"] = (
                (annotations.apply(lambda row: row["prev_speaker_type"] in interactants[row['speaker_type']], axis=1))
                &
                (annotations['iti'] < max_interval)
                &
                (annotations['delay'] >= min_delay)
        )

        return annotations['is_CT'].sum()
    else:
        return 0


peak_simple_CTC = metricFunction(set(), {"speaker_type"})(peak_hour_metric()(simple_CTC))
simple_CTC_ph = metricFunction(set(), {"speaker_type"})(per_hour_metric()(simple_CTC))
simple_CTC = metricFunction(set(), {"speaker_type"})(simple_CTC)
