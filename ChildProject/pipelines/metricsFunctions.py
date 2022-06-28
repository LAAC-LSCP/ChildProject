import pandas as pd
import ast
"""
This file lists all the metrics functions commonly used.
New metrics can be added by defining new functions for the Metrics class to use :
 - New metric functions must have the same arguments:
     - annotations, the metrics pipeline will this
     - duration, also provided by the pipeline
     - kwargs, any keyword argument that is necessary to the metric, those will have to be provided in the metrics_list dataframe to the Metrics class
 - They must return name,value . Name being the default name to attribute to the metric, value being the metric value itself.
 - to compute the metric, use 
     - annotations, which is a dataframe containing all the annotated segments  to use. It contains the annotation content (https://childproject.readthedocs.io/en/latest/format.html#id10) joined with the annotation index info (https://childproject.readthedocs.io/en/latest/format.html#id11) as well as any column that was requested to be added to the results by the user using --child-cols or --rec-cols (eg --child-cols child_dob,languages will make columns 'child_dob' and 'languaes' available)
     - duration which is the duration of audio annotated in milliseconds
     - kwargs, whatever parameter you chose to pass to the function (except 'name', 'callable', 'set' which can not be used)
 - to improve user experience, raise errors and print precise messages about what is wrong.
"""

#error message in case of missing columns in annotations
MISSING_COLUMNS = 'The given set {} does not have the required column {} for computing the {} metric'

def voc_speaker_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """metric calculating number of vocalizations per hour for a given speaker type 
    """
    if 'speaker' not in kwargs : raise ValueError('voc_speaker_ph metric needs an argument <speaker>.')
    name = "voc_{}_ph".format(kwargs["speaker"].lower())
    required_columns = {"speaker_type"}
    if annotations.shape[0]:
        for col in required_columns : 
            if col not in annotations.columns : raise ValueError(MISSING_COLUMNS.format(annotations['set'],col,'voc_speaker_ph'))
        value = annotations[annotations["speaker_type"]== kwargs["speaker"]].shape[0] * (3600000 / duration)
    else:
        value = None
    return name, value
    
def voc_dur_speaker_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """metric calculating number of vocalizations per hour for a given speaker type 
    """
    if 'speaker' not in kwargs : raise ValueError('voc_dur_speaker_ph metric needs an argument <speaker>.')
    name = "voc_dur_{}_ph".format(kwargs["speaker"].lower())
    required_columns = {"speaker_type",'duration'}
    if annotations.shape[0]:
        for col in required_columns : 
            if col not in annotations.columns : raise ValueError(MISSING_COLUMNS.format(annotations['set'],col,'voc_dur_speaker_ph'))
        value = annotations[annotations["speaker_type"]== kwargs["speaker"]]["duration"].sum() * (3600000 / duration)
    else:
        value = None
    return name,value

def avg_voc_dur_speaker(annotations: pd.DataFrame, duration: int, **kwargs):
    """metric calculating the average duration for vocalizations for a given speaker type 
    """
    if 'speaker' not in kwargs : raise ValueError('avg_voc_dur_speaker_ph metric needs an argument <speaker>.')
    name = "avg_voc_dur_{}".format(kwargs["speaker"].lower())
    required_columns = {"speaker_type",'duration'}
    if annotations.shape[0]: 
        for col in required_columns : 
            if col not in annotations.columns : raise ValueError(MISSING_COLUMNS.format(annotations['set'],col,'voc_speaker_ph'))
        value = annotations[annotations["speaker_type"]== kwargs["speaker"]]["duration"].mean()
    else:
        value = None
    return name,value

def wc_speaker_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """metric calculating the number of words per hour for a given speaker type 
    """
    if 'speaker' not in kwargs : raise ValueError('wc_speaker_ph metric needs an argument <speaker>.')
    name = "wc_{}_ph".format(kwargs["speaker"].lower())
    required_columns = {"speaker_type",'words'}
    if annotations.shape[0]: 
        for col in required_columns : 
            if col not in annotations.columns : raise ValueError(MISSING_COLUMNS.format(annotations['set'],col,'voc_speaker_ph'))
        value = annotations[annotations["speaker_type"]== kwargs["speaker"]]["words"].sum() * (3600000 / duration)
    else:
        value = None
    return name,value

def sc_speaker_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """metric calculating the number of syllables per hour for a given speaker type 
    """
    if 'speaker' not in kwargs : raise ValueError('sc_speaker_ph metric needs an argument <speaker>.')
    name = "sc_{}_ph".format(kwargs["speaker"].lower())
    required_columns = {"speaker_type",'syllables'}
    if annotations.shape[0]: 
        for col in required_columns : 
            if col not in annotations.columns : raise ValueError(MISSING_COLUMNS.format(annotations['set'],col,'voc_speaker_ph'))
        value = annotations[annotations["speaker_type"]== kwargs["speaker"]]["syllables"].sum() * (3600000 / duration)
    else:
        value = None
    return name,value

def pc_speaker_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """metric calculating the number of phonemes per hour for a given speaker type 
    """
    if 'speaker' not in kwargs : raise ValueError('pc_speaker_ph metric needs an argument <speaker>.')
    name = "pc_{}_ph".format(kwargs["speaker"].lower())
    required_columns = {"speaker_type",'phonemes'}
    if annotations.shape[0]:
        for col in required_columns : 
            if col not in annotations.columns : raise ValueError(MISSING_COLUMNS.format(annotations['set'],col,'voc_speaker_ph'))
        value = annotations[annotations["speaker_type"]== kwargs["speaker"]]["phonemes"].sum() * (3600000 / duration)
    else:
        value = None
    return name,value

def wc_adu_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """metric calculating the number of words per hour for all speakers
    """
    name = "wc_adu_ph"
    required_columns = {'words'}
    if annotations.shape[0]:
        for col in required_columns : 
            if col not in annotations.columns : raise ValueError(MISSING_COLUMNS.format(annotations['set'],col,'voc_speaker_ph'))
        value = annotations["words"].sum() * (3600000 / duration)
    else:
        value = None
    return name,value

def sc_adu_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """metric calculating the number of syllables per hour for all speakers
    """
    name = "sc_adu_ph"
    required_columns = {'syllables'}
    if annotations.shape[0]:
        for col in required_columns : 
            if col not in annotations.columns : raise ValueError(MISSING_COLUMNS.format(annotations['set'],col,'voc_speaker_ph'))
        value = annotations["syllables"].sum() * (3600000 / duration)
    else:
        value = None
    return name,value

def pc_adu_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """metric calculating the number of phonemes per hour for all speakers
    """
    name = "pc_adu_ph"
    required_columns = {"phonemes"}
    if annotations.shape[0]:
        for col in required_columns : 
            if col not in annotations.columns : raise ValueError(MISSING_COLUMNS.format(annotations['set'],col,'voc_speaker_ph'))
        value = annotations["phonemes"].sum() * (3600000 / duration)
    else:
        value = None
    return name,value

def cry_voc_speaker_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """metric calculating the number of cries per hour for CHI (based on vcm_type)
    """
    if 'speaker' not in kwargs : raise ValueError('cry_voc_speaker_ph metric needs an argument <speaker>.')
    name = "cry_voc_{}_ph".format(kwargs["speaker"].lower())
    required_columns = {"speaker_type",'vcm_type'}
    if annotations.shape[0]:
        for col in required_columns : 
            if col not in annotations.columns : raise ValueError(MISSING_COLUMNS.format(annotations['set'],col,'voc_speaker_ph'))
        value = annotations.loc[(annotations["speaker_type"]== kwargs["speaker"]) & (annotations["vcm_type"]== "Y")].shape[0] * (3600000 / duration)
    else:
        value = None
    return name,value

def cry_voc_dur_speaker_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """metric calculating the duration of cries per hour for CHI (based on vcm_type)
    """
    if 'speaker' not in kwargs : raise ValueError('cry_voc_dur_speaker_ph metric needs an argument <speaker>.')
    name = "cry_voc_dur_{}_ph".format(kwargs["speaker"].lower())
    required_columns = {"speaker_type",'vcm_type','duration'}
    if annotations.shape[0]:
        for col in required_columns : 
            if col not in annotations.columns : raise ValueError(MISSING_COLUMNS.format(annotations['set'],col,'voc_speaker_ph'))
        value = annotations.loc[(annotations["speaker_type"]== kwargs["speaker"]) & (annotations["vcm_type"]== "Y")]["duration"].sum() * (3600000 / duration)
    else:
        value = None
    return name,value

def avg_cry_voc_dur_speaker(annotations: pd.DataFrame, duration: int, **kwargs):
    """metric calculating the average duration of cries for CHI (based on vcm_type)
    """
    if 'speaker' not in kwargs : raise ValueError('avg_cry_voc_dur_speaker metric needs an argument <speaker>.')
    name = "avg_cry_voc_dur_{}".format(kwargs["speaker"].lower())
    required_columns = {"speaker_type",'vcm_type','duration'}
    if annotations.shape[0]:
        for col in required_columns : 
            if col not in annotations.columns : raise ValueError(MISSING_COLUMNS.format(annotations['set'],col,'voc_speaker_ph'))
        value = annotations.loc[(annotations["speaker_type"]== kwargs["speaker"]) & (annotations["vcm_type"]== "Y")]["duration"].mean() * (3600000 / duration)
        if pd.isnull(value) : value = 0
    else:
        value = None
    return name,value

def can_voc_speaker_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """metric calculating the number of canonical vocalizations per hour for CHI (based on vcm_type)
    """
    if 'speaker' not in kwargs : raise ValueError('can_voc_speaker_ph metric needs an argument <speaker>.')
    name = "can_voc_{}_ph".format(kwargs["speaker"].lower())
    required_columns = {"speaker_type",'vcm_type'}
    if annotations.shape[0]:
        for col in required_columns : 
            if col not in annotations.columns : raise ValueError(MISSING_COLUMNS.format(annotations['set'],col,'voc_speaker_ph'))
        value = annotations.loc[(annotations["speaker_type"]== kwargs["speaker"]) & (annotations["vcm_type"]== "C")].shape[0] * (3600000 / duration)
    else:
        value = None
    return name,value

def can_voc_dur_speaker_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """metric calculating the duration of canonical vocalizations per hour for CHI (based on vcm_type)
    """
    if 'speaker' not in kwargs : raise ValueError('can_voc_dur_speaker_ph metric needs an argument <speaker>.')
    name = "can_voc_dur_{}_ph".format(kwargs["speaker"].lower())
    required_columns = {"speaker_type",'vcm_type','duration'}
    if annotations.shape[0]:
        for col in required_columns : 
            if col not in annotations.columns : raise ValueError(MISSING_COLUMNS.format(annotations['set'],col,'voc_speaker_ph'))
        value = annotations.loc[(annotations["speaker_type"]== kwargs["speaker"]) & (annotations["vcm_type"]== "C")]["duration"].sum() * (3600000 / duration)
    else:
        value = None
    return name,value

def avg_can_voc_dur_speaker(annotations: pd.DataFrame, duration: int, **kwargs):
    """metric calculating the average duration of canonical vocalizations for CHI (based on vcm_type)
    """
    if 'speaker' not in kwargs : raise ValueError('can_voc_dur_speaker_ph metric needs an argument <speaker>.')
    name = "avg_can_voc_dur_{}".format(kwargs["speaker"].lower())
    required_columns = {"speaker_type",'vcm_type','duration'}
    if annotations.shape[0]:
        for col in required_columns : 
            if col not in annotations.columns : raise ValueError(MISSING_COLUMNS.format(annotations['set'],col,'voc_speaker_ph'))
        value = annotations.loc[(annotations["speaker_type"]== kwargs["speaker"]) & (annotations["vcm_type"]== "C")]["duration"].mean() * (3600000 / duration)
        if pd.isnull(value) : value = 0
    else:
        value = None
    return name,value

def non_can_voc_speaker_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """metric calculating the number of non canonical vocalizations per hour for CHI (based on vcm_type)
    """
    if 'speaker' not in kwargs : raise ValueError('can_voc_dur_speaker_ph metric needs an argument <speaker>.')
    name = "non_can_voc_{}_ph".format(kwargs["speaker"].lower())
    required_columns = {"speaker_type",'vcm_type'}
    if annotations.shape[0]:
        for col in required_columns : 
            if col not in annotations.columns : raise ValueError(MISSING_COLUMNS.format(annotations['set'],col,'voc_speaker_ph'))
        value = annotations.loc[(annotations["speaker_type"]== kwargs["speaker"]) & (annotations["vcm_type"]== "N")].shape[0] * (3600000 / duration)
    else:
        value = None
    return name,value

def non_can_voc_dur_speaker_ph(annotations: pd.DataFrame, duration: int, **kwargs):
    """metric calculating the duration of non canonical vocalizations per hour for CHI (based on vcm_type)
    """
    if 'speaker' not in kwargs : raise ValueError('can_voc_dur_speaker_ph metric needs an argument <speaker>.')
    name = "non_can_voc_dur_{}_ph".format(kwargs["speaker"].lower())
    required_columns = {"speaker_type",'vcm_type','duration'}
    if annotations.shape[0]:
        for col in required_columns : 
            if col not in annotations.columns : raise ValueError(MISSING_COLUMNS.format(annotations['set'],col,'voc_speaker_ph'))
        value = annotations.loc[(annotations["speaker_type"]== kwargs["speaker"]) & (annotations["vcm_type"]== "N")]["duration"].sum() * (3600000 / duration)
    else:
        value = None
    return name,value

def avg_non_can_voc_dur_speaker(annotations: pd.DataFrame, duration: int, **kwargs):
    """metric calculating the average duration of non canonical vocalizations for CHI (based on vcm_type)
    """
    if 'speaker' not in kwargs : raise ValueError('can_voc_dur_speaker_ph metric needs an argument <speaker>.')
    name = "avg_non_can_voc_dur_{}".format(kwargs["speaker"].lower())
    required_columns = {"speaker_type",'vcm_type','duration'}
    if annotations.shape[0]:
        for col in required_columns : 
            if col not in annotations.columns : raise ValueError(MISSING_COLUMNS.format(annotations['set'],col,'voc_speaker_ph'))
        value = annotations.loc[(annotations["speaker_type"]== kwargs["speaker"]) & (annotations["vcm_type"]== "N")]["duration"].mean() * (3600000 / duration)
        if pd.isnull(value) : value = 0
    else:
        value = None
    return name,value

def lp_n(annotations: pd.DataFrame, duration: int, **kwargs):
    """metric calculating the linguistic proportion on the number of vocalizations for CHI (based on vcm_type or [cries,vfxs,utterances_count] if vcm_type does not exist)
    """
    name = "lp_n"
    if annotations.shape[0]: 
        if "vcm_type" in annotations.columns:
            speech_voc = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"].isin(["N","C"]))].shape[0]
            cry_voc = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "Y")].shape[0]
            value = speech_voc / (speech_voc + cry_voc)
        elif set(["cries","vfxs","utterances_count"]).issubset(annotations.columns):
            annotations = annotations[annotations["speaker_type"] == "CHI"]
            cries = annotations["cries"].apply(lambda x: len(ast.literal_eval(x))).sum()
            vfxs = annotations["vfxs"].apply(lambda x: len(ast.literal_eval(x))).sum()
            utterances = annotations["utterances_count"].sum()
            value = utterances / (utterances + cries + vfxs)
        else:
            raise ValueError("the given set does not have the neccessary columns for this metric, choose a set that contains either [vcm_type] or [cries,vfxs,utterances_count]")
    else:
        value = None
    return name,value

def cp_n(annotations: pd.DataFrame, duration: int, **kwargs):
    """metric calculating the canonical proportion on the number of vocalizations for CHI (based on vcm_type)
    """
    name = "cp_n"
    required_columns = {"speaker_type",'vcm_type'}
    if annotations.shape[0]:
        for col in required_columns : 
            if col not in annotations.columns : raise ValueError(MISSING_COLUMNS.format(annotations['set'],col,'voc_speaker_ph'))
        speech_voc = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"].isin(["N","C"]))].shape[0]
        can_voc = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "C")].shape[0]
        value = can_voc / speech_voc
    else:
        value = None
    return name,value

def lp_dur(annotations: pd.DataFrame, duration: int, **kwargs):
    """metric calculating the linguistic proportion on the duration of vocalizations for CHI (based on vcm_type or [child_cry_vfxs_len,utterances_length] if vcm_type does not exist)
    """
    name = "lp_dur"
    if annotations.shape[0]: 
        if "vcm_type" in annotations.columns:
            speech_dur = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"].isin(["N","C"]))]["duration"].sum()
            cry_dur = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "Y")]["duration"].sum()
            value = speech_dur / (speech_dur + cry_dur)
        elif set(["child_cry_vfx_len","utterances_length"]).issubset(annotations.columns):
            annotations = annotations[annotations["speaker_type"] == "CHI"]
            value = annotations["utterances_length"].sum() / (
                annotations["child_cry_vfx_len"].sum() + annotations["utterances_length"].sum() )
        else:
            raise ValueError("the {} set does not have the neccessary columns for this metric, choose a set that contains either [vcm_type] or [child_cry_vfx_len,utterances_length]")
    else:
        value = None
    return name,value

def cp_dur(annotations: pd.DataFrame, duration: int, **kwargs):
    """metric calculating the canonical proportion on the number of vocalizations for CHI (based on vcm_type)
    """
    name = "cp_dur"
    required_columns = {"speaker_type",'vcm_type','duration'}
    if annotations.shape[0]:
        for col in required_columns : 
            if col not in annotations.columns : raise ValueError(MISSING_COLUMNS.format(annotations['set'],col,'voc_speaker_ph'))
        speech_dur = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"].isin(["N","C"]))]["duration"].sum()
        can_dur = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "C")]["duration"].sum()
        value = can_dur / speech_dur
    else:
        value = None
    return name,value