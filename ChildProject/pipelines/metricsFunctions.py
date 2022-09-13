import pandas as pd
import numpy as np
import ast
import re
import functools
"""
This file lists all the metrics functions commonly used.
New metrics can be added by defining new functions for the Metrics class to use :
 - Create a new function using the same arguments (i.e. annotations, duration, **kwargs)
 - Define calculation of the metric with:
     - annotations, which is a dataframe containing all the relevant annotated segments  to use. It contains the annotation content (https://childproject.readthedocs.io/en/latest/format.html#id10) joined with the annotation index info (https://childproject.readthedocs.io/en/latest/format.html#id11) as well as any column that was requested to be added to the results by the user using --child-cols or --rec-cols (eg --child-cols child_dob,languages will make columns 'child_dob' and 'languaes' available)
     - duration which is the duration of audio annotated in milliseconds
     - kwargs, whatever keyword parameter you chose to pass to the function (except 'name', 'callable', 'set' which can not be used). This will need to be given with the list of metrics when called
 - Wrap you function with the 'metricFunction' decorator to make it callable by the pipeline, read metricFunction help for more info
"""

#error message in case of missing columns in annotations
MISSING_COLUMNS = 'The given set <{}> does not have the required column <{}> for computing the {} metric'
    
def metricFunction(args: set, columns: set, emptyValue = 0, name : str = None):
    """Decorator for all metrics functions to make them ready to be called by the pipeline.
    
    :param args: set of required keyword arguments for that function, raise ValueError if were not given
    :type args: set
    :param columns: set of required columns in the dataframe given, missing columns raise ValueError
    :type columns: set
    :param name: default name to use for the metric in the resulting dataframe. Every keyword argument found in the name will be replaced by its value (e.g. 'voc_speaker_ph' uses kwarg 'speaker' so if speaker = 'CHI', name will be 'voc_chi_ph'). if no name is given, the __name__ of the function is used
    :type name: str
    :param emptyValue: value to return when annotations are empty but the unit was annotated (e.g. 0 for counts like voc_speaker_ph , None for proportions like lp_n)
    :return: new function to substitute the metric function
    :rtype: Callable
    """
    def decorator(function):
        @functools.wraps(function)
        def new_func(annotations: pd.DataFrame, duration: int, **kwargs):
            for arg in args:
                if arg not in kwargs : raise ValueError('{} metric needs an argument <{}>'.format(function.__name__,arg))
            metric_name = name
            if not name : metric_name = function.__name__
            metric_name_replaced = metric_name
            #metric_name is the basename used to designate this metric (voc_speaker_ph), metric_name_replaced replaces the values of kwargs
            #found in the name by their values, giving the metric name for that instance only (voc_chi_ph)
            for arg in kwargs:
                metric_name_replaced = re.sub(arg , str(kwargs[arg]).lower(),metric_name_replaced)
            if annotations.shape[0]:
                missing_columns = columns - set(annotations.columns)
                if missing_columns:
                    raise ValueError(MISSING_COLUMNS.format(annotations['set'].iloc[0],missing_columns,metric_name))
                res = function(annotations, duration, **kwargs)
            else: #no annotation for that unit
                res = emptyValue if duration else None #duration != 0 => was annotated but not segments there
            return metric_name_replaced, res
        return new_func
    return decorator
            

@metricFunction({"speaker"},{"speaker_type"}) 
def voc_speaker(annotations: pd.DataFrame, duration: int, **kwargs):
    """number of vocalizations for a given speaker type
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    return annotations[annotations["speaker_type"]== kwargs["speaker"]].shape[0]

@metricFunction({"speaker"},{"speaker_type"}) 
def voc_speaker_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """number of vocalizations per hour for a given speaker type
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    return annotations[annotations["speaker_type"]== kwargs["speaker"]].shape[0] * (3600000 / duration)

@metricFunction({"speaker"},{"speaker_type","duration"})
def voc_dur_speaker_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """total duration of vocalizations by a given speaker type in milliseconds per hour
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    return annotations[annotations["speaker_type"]== kwargs["speaker"]]["duration"].sum() * (3600000 / duration)

@metricFunction({"speaker"},{"speaker_type","duration"},np.nan)
def avg_voc_dur_speaker(annotations: pd.DataFrame, duration: int, **kwargs):
    """average duration in milliseconds of vocalizations for a given speaker type 
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    return annotations[annotations["speaker_type"]== kwargs["speaker"]]["duration"].mean()

@metricFunction({"speaker"},{"speaker_type","words"})
def wc_speaker_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """number of words per hour for a given speaker type
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    return annotations[annotations["speaker_type"]== kwargs["speaker"]]["words"].sum() * (3600000 / duration)

@metricFunction({"speaker"},{"speaker_type","syllables"})
def sc_speaker_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """number of syllables per hour for a given speaker type
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    return annotations[annotations["speaker_type"]== kwargs["speaker"]]["syllables"].sum() * (3600000 / duration)

@metricFunction({"speaker"},{"speaker_type","phonemes"})
def pc_speaker_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """number of phonemes per hour for a given speaker type
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    return annotations[annotations["speaker_type"]== kwargs["speaker"]]["phonemes"].sum() * (3600000 / duration)

@metricFunction({},{"words"})
def wc_adu_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """number of words per hour for all speakers
    
    Required keyword arguments:
    """
    return annotations["words"].sum() * (3600000 / duration)

@metricFunction({},{"syllables"})
def sc_adu_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """number of syllables per hour for all speakers
    
    Required keyword arguments:
    """
    return annotations["syllables"].sum() * (3600000 / duration)

@metricFunction({},{"phonemes"})
def pc_adu_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """number of phonemes per hour for all speakers
    
    Required keyword arguments:
    """
    return annotations["phonemes"].sum() * (3600000 / duration)
    
@metricFunction({"speaker"},{"speaker_type","vcm_type"})
def cry_voc_speaker_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """number of cry vocalizations per hour for a given speaker (based on vcm_type)
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    return annotations.loc[(annotations["speaker_type"]== kwargs["speaker"]) & (annotations["vcm_type"]== "Y")].shape[0] * (3600000 / duration)

@metricFunction({"speaker"},{"speaker_type","vcm_type","duration"})
def cry_voc_dur_speaker_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """total duration of cry vocalizations by a given speaker type in milliseconds per hour (based on vcm_type)
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    return annotations.loc[(annotations["speaker_type"]== kwargs["speaker"]) & (annotations["vcm_type"]== "Y")]["duration"].sum() * (3600000 / duration)

@metricFunction({"speaker"},{"speaker_type","vcm_type","duration"},np.nan)
def avg_cry_voc_dur_speaker(annotations: pd.DataFrame, duration: int, **kwargs):
    """average duration of cry vocalizations by a given speaker type (based on vcm_type)
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    value = annotations.loc[(annotations["speaker_type"]== kwargs["speaker"]) & (annotations["vcm_type"]== "Y")]["duration"].mean()
    if pd.isnull(value) : value = 0
    return value

@metricFunction({"speaker"},{"speaker_type","vcm_type"})
def can_voc_speaker_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """number of canonical vocalizations per hour for a given speaker type (based on vcm_type)
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    return annotations.loc[(annotations["speaker_type"]== kwargs["speaker"]) & (annotations["vcm_type"]== "C")].shape[0] * (3600000 / duration)
    
@metricFunction({"speaker"},{"speaker_type","vcm_type","duration"})
def can_voc_dur_speaker_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """total duration of canonical vocalizations by a given speaker type in milliseconds per hour (based on vcm_type)
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    return annotations.loc[(annotations["speaker_type"]== kwargs["speaker"]) & (annotations["vcm_type"]== "C")]["duration"].sum() * (3600000 / duration)

@metricFunction({"speaker"},{"speaker_type","vcm_type","duration"},np.nan)
def avg_can_voc_dur_speaker(annotations: pd.DataFrame, duration: int, **kwargs):
    """average duration of canonical vocalizations for a given speaker type (based on vcm_type)
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    value =  annotations.loc[(annotations["speaker_type"]== kwargs["speaker"]) & (annotations["vcm_type"]== "C")]["duration"].mean()
    if pd.isnull(value) : value = 0
    return value

@metricFunction({"speaker"},{"speaker_type","vcm_type"})
def non_can_voc_speaker_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """number of non canonical vocalizations per hour for a given speaker type (based on vcm_type)
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    return annotations.loc[(annotations["speaker_type"]== kwargs["speaker"]) & (annotations["vcm_type"]== "N")].shape[0] * (3600000 / duration)

@metricFunction({"speaker"},{"speaker_type","vcm_type","duration"})
def non_can_voc_dur_speaker_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """total duration of non canonical vocalizations by a given speaker type in milliseconds per hour (based on vcm_type)
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    return annotations.loc[(annotations["speaker_type"]== kwargs["speaker"]) & (annotations["vcm_type"]== "N")]["duration"].sum() * (3600000 / duration)

@metricFunction({"speaker"},{"speaker_type","vcm_type","duration"},np.nan)
def avg_non_can_voc_dur_speaker(annotations: pd.DataFrame, duration: int, **kwargs):
    """average duration of non canonical vocalizations for a given speaker type (based on vcm_type)
    
    Required keyword arguments:
        - speaker : speaker_type to use
    """
    value = annotations.loc[(annotations["speaker_type"]== kwargs["speaker"]) & (annotations["vcm_type"]== "N")]["duration"].mean()
    if pd.isnull(value) : value = 0
    return value

@metricFunction(set(),set(),np.nan)
def lp_n(annotations: pd.DataFrame, duration: int, **kwargs):
    """linguistic proportion on the number of vocalizations for CHI (based on vcm_type or [cries,vfxs,utterances_count] if vcm_type does not exist)
    
    Required keyword arguments:
    """
    if set(["cries","vfxs","utterances_count"]).issubset(annotations.columns):
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
        speech_voc = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"].isin(["N","C"]))].shape[0]
        cry_voc = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "Y")].shape[0]
        total = speech_voc + cry_voc
        if total:
            value = speech_voc / total
        else:
            value = np.nan
    else:
        raise ValueError("the given set does not have the neccessary columns for this metric, choose a set that contains either [vcm_type] or [cries,vfxs,utterances_count]")
    return value

@metricFunction(set(),{"speaker_type","vcm_type"},np.nan)
def cp_n(annotations: pd.DataFrame, duration: int, **kwargs):
    """canonical proportion on the number of vocalizations for CHI (based on vcm_type)
    
    Required keyword arguments:
    """
    speech_voc = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"].isin(["N","C"]))].shape[0]
    can_voc = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "C")].shape[0]
    if speech_voc:
        value = can_voc / speech_voc
    else:
        value = np.nan
    return value
    
@metricFunction(set(),set(),np.nan)
def lp_dur(annotations: pd.DataFrame, duration: int, **kwargs):
    """linguistic proportion on the duration of vocalizations for CHI (based on vcm_type or [child_cry_vfxs_len,utterances_length] if vcm_type does not exist)
    
    Required keyword arguments:
    """
    if set(["child_cry_vfx_len","utterances_length"]).issubset(annotations.columns):
        annotations = annotations[annotations["speaker_type"] == "CHI"]
        utter_len = annotations["utterances_length"].sum()
        total = annotations["child_cry_vfx_len"].sum() + utter_len
        if total:
            value = utter_len / total
        else:
            value = np.nan
    elif "vcm_type" in annotations.columns:
        speech_dur = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"].isin(["N","C"]))]["duration"].sum()
        cry_dur = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "Y")]["duration"].sum()
        total = speech_dur + cry_dur
        if total:
            value = speech_dur / total
        else:
            value = np.nan
    else:
        raise ValueError("the {} set does not have the neccessary columns for this metric, choose a set that contains either [vcm_type] or [child_cry_vfx_len,utterances_length]")
    return value

@metricFunction(set(),{"speaker_type","vcm_type","duration"},np.nan)
def cp_dur(annotations: pd.DataFrame, duration: int, **kwargs):
    """canonical proportion on the number of vocalizations for CHI (based on vcm_type)
    
    Required keyword arguments:
    """
    speech_dur = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"].isin(["N","C"]))]["duration"].sum()
    can_dur = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "C")]["duration"].sum()
    if speech_dur:
        value = can_dur / speech_dur
    else:
        value = np.nan
    return value

@metricFunction(set(),{"utterances_count"}) 
def lena_CVC(annotations: pd.DataFrame, duration: int, **kwargs):
    """number of child vocalizations according to LENA's extraction
    
    Required keyword arguments:
    """
    return annotations["utterances_count"].sum()
    
@metricFunction(set(),{"lena_conv_turn_type"}) 
def lena_CTC(annotations: pd.DataFrame, duration: int, **kwargs):
    """number of conversational turn counts according to LENA's extraction
    
    Required keyword arguments:
    """
    conv_types = {'TIMR', 'TIFR'}
    return annotations[annotations["lena_conv_turn_type"].isin(conv_types)].shape[0]