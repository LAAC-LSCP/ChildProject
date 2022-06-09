import pandas as pd
import ast

# This file lists all the metrics functions commonly used.
# New metrics ccan be added by defining new functions for the Metrics class to use


#########################################
# TODO check presence of correct arguments for each metric, check given annotation set has the wanted columns
#
#########################################
    
def voc_speaker_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating number of vocalizations per hour for a given speaker type 
    """
    name = "voc_{}_ph".format(arguments["speaker"].lower())
    if annotations.shape[0]:      
        value = annotations[annotations["speaker_type"]== arguments["speaker"]].shape[0] * (3600 / duration)
    else:
        value = None
    return name, value
    
def voc_dur_speaker_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating number of vocalizations per hour for a given speaker type 
    """
    name = "voc_dur_{}_ph".format(arguments["speaker"].lower())
    if annotations.shape[0]: 
        value = annotations[annotations["speaker_type"]== arguments["speaker"]]["duration"].sum() * (3600 / duration)
    else:
        value = None
    return name,value

def avg_voc_dur_speaker(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the average duration for vocalizations for a given speaker type 
    """
    name = "avg_voc_dur_{}".format(arguments["speaker"].lower())
    if annotations.shape[0]: 
        value = annotations[annotations["speaker_type"]== arguments["speaker"]]["duration"].mean()
    else:
        value = None
    return name,value

def wc_speaker_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the number of words per hour for a given speaker type 
    """
    name = "wc_{}_ph".format(arguments["speaker"].lower())
    if annotations.shape[0]: 
        value = annotations[annotations["speaker_type"]== arguments["speaker"]]["words"].sum() * (3600 / duration)
    else:
        value = None
    return name,value

def sc_speaker_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the number of syllables per hour for a given speaker type 
    """
    name = "sc_{}_ph".format(arguments["speaker"].lower())
    if annotations.shape[0]: 
        value = annotations[annotations["speaker_type"]== arguments["speaker"]]["syllables"].sum() * (3600 / duration)
    else:
        value = None
    return name,value

def pc_speaker_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the number of phonemes per hour for a given speaker type 
    """
    name = "pc_{}_ph".format(arguments["speaker"].lower())
    if annotations.shape[0]: 
        value = annotations[annotations["speaker_type"]== arguments["speaker"]]["phonemes"].sum() * (3600 / duration)
    else:
        value = None
    return name,value

def wc_adu_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the number of words per hour for all speakers
    """
    name = "wc_adu_ph"
    if annotations.shape[0]: 
        value = annotations["words"].sum() * (3600 / duration)
    else:
        value = None
    return name,value

def sc_adu_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the number of syllables per hour for all speakers
    """
    name = "sc_adu_ph"
    if annotations.shape[0]: 
        value = annotations["syllables"].sum() * (3600 / duration)
    else:
        value = None
    return name,value

def pc_adu_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the number of phonemes per hour for all speakers
    """
    name = "pc_adu_ph"
    if annotations.shape[0]: 
        value = annotations["phonemes"].sum() * (3600 / duration)
    else:
        value = None
    return name,value

def cry_voc_chi_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the number of cries per hour for CHI (based on vcm_type)
    """
    name = "cry_voc_chi_ph"
    if annotations.shape[0]: 
        value = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "Y")].shape[0] * (3600 / duration)
    else:
        value = None
    return name,value

def cry_voc_dur_chi_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the duration of cries per hour for CHI (based on vcm_type)
    """
    name = "cry_voc_dur_chi_ph"
    if annotations.shape[0]: 
        value = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "Y")]["duration"].sum() * (3600 / duration)
    else:
        value = None
    return name,value

def avg_cry_voc_dur_chi(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the average duration of cries for CHI (based on vcm_type)
    """
    name = "avg_cry_voc_dur_chi"
    if annotations.shape[0]: 
        value = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "Y")]["duration"].mean() * (3600 / duration)
        if pd.isnull(value) : value = 0
    else:
        value = None
    return name,value

def can_voc_chi_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the number of canonical vocalizations per hour for CHI (based on vcm_type)
    """
    name = "can_voc_chi_ph"
    if annotations.shape[0]: 
        value = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "C")].shape[0] * (3600 / duration)
    else:
        value = None
    return name,value

def can_voc_dur_chi_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the duration of canonical vocalizations per hour for CHI (based on vcm_type)
    """
    name = "can_voc_dur_chi_ph"
    if annotations.shape[0]: 
        value = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "C")]["duration"].sum() * (3600 / duration)
    else:
        value = None
    return name,value

def avg_can_voc_dur_chi(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the average duration of canonical vocalizations for CHI (based on vcm_type)
    """
    name = "avg_can_voc_dur_chi"
    if annotations.shape[0]: 
        value = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "C")]["duration"].mean() * (3600 / duration)
        if pd.isnull(value) : value = 0
    else:
        value = None
    return name,value

def non_can_voc_chi_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the number of non canonical vocalizations per hour for CHI (based on vcm_type)
    """
    name = "non_can_voc_chi_ph"
    if annotations.shape[0]: 
        value = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "N")].shape[0] * (3600 / duration)
    else:
        value = None
    return name,value

def non_can_voc_dur_chi_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the duration of non canonical vocalizations per hour for CHI (based on vcm_type)
    """
    name = "non_can_voc_dur_chi_ph"
    if annotations.shape[0]: 
        value = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "N")]["duration"].sum() * (3600 / duration)
    else:
        value = None
    return name,value

def avg_non_can_voc_dur_chi(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the average duration of non canonical vocalizations for CHI (based on vcm_type)
    """
    name = "avg_non_can_voc_dur_chi"
    if annotations.shape[0]: 
        value = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "N")]["duration"].mean() * (3600 / duration)
        if pd.isnull(value) : value = 0
    else:
        value = None
    return name,value

def lp_n(annotations: pd.DataFrame, duration: int, arguments: dict = None):
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

def cp_n(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the canonical proportion on the number of vocalizations for CHI (based on vcm_type)
    """
    name = "cp_n"
    if annotations.shape[0]: 
        speech_voc = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"].isin(["N","C"]))].shape[0]
        can_voc = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "C")].shape[0]
        value = can_voc / speech_voc
    else:
        value = None
    return name,value

def lp_dur(annotations: pd.DataFrame, duration: int, arguments: dict = None):
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

def cp_dur(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the canonical proportion on the number of vocalizations for CHI (based on vcm_type)
    """
    name = "cp_dur"
    if annotations.shape[0]: 
        speech_dur = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"].isin(["N","C"]))]["duration"].sum()
        can_dur = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "C")]["duration"].sum()
        value = can_dur / speech_dur
    else:
        value = None
    return name,value