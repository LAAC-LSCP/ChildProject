from abc import ABC, abstractmethod
import argparse
import datetime
import multiprocessing as mp
import numpy as np
import pandas as pd
from typing import Union, List
import datetime
import ast

import ChildProject
from ChildProject.pipelines.pipeline import Pipeline

pipelines = {}

#########################################
# TODO check presence of correct arguments for each metric, check given annotation set has the wanted columns
#########################################
    
def voc_speaker_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating number of vocalizations per hour for a given speaker type 
    """
    name = "voc_{}_ph".format(arguments["speaker"].lower())
    value = annotations[annotations["speaker_type"]== arguments["speaker"]].shape[0] * (3600 / duration)
    return name, value
    
def voc_dur_speaker_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating number of vocalizations per hour for a given speaker type 
    """
    name = "voc_dur_{}_ph".format(arguments["speaker"].lower())
    value = annotations[annotations["speaker_type"]== arguments["speaker"]]["duration"].sum() * (3600 / duration)
    return name,value

def avg_voc_dur_speaker(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the average duration for vocalizations for a given speaker type 
    """
    name = "avg_voc_dur_{}".format(arguments["speaker"].lower())
    value = annotations[annotations["speaker_type"]== arguments["speaker"]]["duration"].mean()
    return name,value

def wc_speaker_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the number of words per hour for a given speaker type 
    """
    name = "wc_{}_ph".format(arguments["speaker"].lower())
    value = annotations[annotations["speaker_type"]== arguments["speaker"]]["words"].sum() * (3600 / duration)
    return name,value

def sc_speaker_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the number of syllables per hour for a given speaker type 
    """
    name = "sc_{}_ph".format(arguments["speaker"].lower())
    value = annotations[annotations["speaker_type"]== arguments["speaker"]]["syllables"].sum() * (3600 / duration)
    return name,value

def pc_speaker_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the number of phonemes per hour for a given speaker type 
    """
    name = "pc_{}_ph".format(arguments["speaker"].lower())
    value = annotations[annotations["speaker_type"]== arguments["speaker"]]["phonemes"].sum() * (3600 / duration)
    return name,value

def wc_adu_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the number of words per hour for all speakers
    """
    name = "wc_adu_ph"
    value = annotations["words"].sum() * (3600 / duration)
    return name,value

def sc_adu_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the number of syllables per hour for all speakers
    """
    name = "sc_adu_ph"
    value = annotations["syllables"].sum() * (3600 / duration)
    return name,value

def pc_adu_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the number of phonemes per hour for all speakers
    """
    name = "pc_adu_ph"
    value = annotations["phonemes"].sum() * (3600 / duration)
    return name,value

def cry_voc_chi_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the number of cries per hour for CHI (based on vcm_type)
    """
    name = "cry_voc_chi_ph"
    value = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "Y")].shape[0] * (3600 / duration)
    return name,value

def cry_voc_dur_chi_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the duration of cries per hour for CHI (based on vcm_type)
    """
    name = "cry_voc_dur_chi_ph"
    value = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "Y")]["duration"].sum() * (3600 / duration)
    return name,value

def avg_cry_voc_dur_chi(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the average duration of cries for CHI (based on vcm_type)
    """
    name = "avg_cry_voc_dur_chi"
    value = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "Y")]["duration"].mean() * (3600 / duration)
    if pd.isnull(value) : value = 0
    return name,value

def can_voc_chi_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the number of canonical vocalizations per hour for CHI (based on vcm_type)
    """
    name = "can_voc_chi_ph"
    value = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "C")].shape[0] * (3600 / duration)
    return name,value

def can_voc_dur_chi_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the duration of canonical vocalizations per hour for CHI (based on vcm_type)
    """
    name = "can_voc_dur_chi_ph"
    value = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "C")]["duration"].sum() * (3600 / duration)
    return name,value

def avg_can_voc_dur_chi(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the average duration of canonical vocalizations for CHI (based on vcm_type)
    """
    name = "avg_can_voc_dur_chi"
    value = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "C")]["duration"].mean() * (3600 / duration)
    if pd.isnull(value) : value = 0
    return name,value

def non_can_voc_chi_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the number of non canonical vocalizations per hour for CHI (based on vcm_type)
    """
    name = "non_can_voc_chi_ph"
    value = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "N")].shape[0] * (3600 / duration)
    return name,value

def non_can_voc_dur_chi_ph(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the duration of non canonical vocalizations per hour for CHI (based on vcm_type)
    """
    name = "non_can_voc_dur_chi_ph"
    value = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "N")]["duration"].sum() * (3600 / duration)
    return name,value

def avg_non_can_voc_dur_chi(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the average duration of non canonical vocalizations for CHI (based on vcm_type)
    """
    name = "avg_non_can_voc_dur_chi"
    value = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "N")]["duration"].mean() * (3600 / duration)
    if pd.isnull(value) : value = 0
    return name,value

def lp_n(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the linguistic proportion on the number of vocalizations for CHI (based on vcm_type or [cries,vfxs,utterances_count] if vcm_type does not exist)
    """
    name = "lp_n"
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
        raise ValueError("the {} set does not have the neccessary columns for this metric, choose a set that contains either [vcm_type] or [cries,vfxs,utterances_count]")
    return name,value

def cp_n(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the canonical proportion on the number of vocalizations for CHI (based on vcm_type)
    """
    name = "cp_n"
    speech_voc = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"].isin(["N","C"]))].shape[0]
    can_voc = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "C")].shape[0]
    value = can_voc / speech_voc
    return name,value

def lp_dur(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the linguistic proportion on the duration of vocalizations for CHI (based on vcm_type or [child_cry_vfxs_len,utterances_length] if vcm_type does not exist)
    """
    name = "lp_dur"
    if "vcm_type" in annotations.columns:
        speech_dur = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"].isin(["N","C"]))]["duration"].sum()
        cry_dur = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "Y")]["duration"].sum()
        value = speech_dur / (speech_dur + cry_dur)
    elif set(["child_cry_vfxs_len","utterances_length"]).issubset(annotations.columns):
        annotations = annotations[annotations["speaker_type"] == "CHI"]
        value = annotations["utterances_length"].sum() / (
            annotations["child_cry_vfx_len"].sum() + annotations["utterances_length"].sum() )
    else:
        raise ValueError("the {} set does not have the neccessary columns for this metric, choose a set that contains either [vcm_type] or [child_cry_vfxs_len,utterances_length]")
    return name,value

def cp_dur(annotations: pd.DataFrame, duration: int, arguments: dict = None):
    """metric calculating the canonical proportion on the number of vocalizations for CHI (based on vcm_type)
    """
    name = "cp_dur"
    speech_dur = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"].isin(["N","C"]))]["duration"].sum()
    can_dur = annotations.loc[(annotations["speaker_type"]== "CHI") & (annotations["vcm_type"]== "C")]["duration"].sum()
    value = can_dur / speech_dur
    return name,value

class Metrics(ABC):
    def __init__(
        self,
        project: ChildProject.projects.ChildProject,
        metrics_list: pd.DataFrame,
        by: str = "recording_filename",
        recordings: Union[str, List[str], pd.DataFrame] = None,
        from_time: str = None,
        to_time: str = None,
        rec_cols: list = [],
        child_cols: list = [],
        period: str = None,
        threads: int = 1,
    ):

        self.project = project
        self.am = ChildProject.annotations.AnnotationManager(self.project)
        self.threads = int(threads)
        
        ##TODO check validity of metrics_list (dataframe with callable, set, arguments, name?)
        metrics_list.sort_values(by="set",inplace=True)
        self.metrics_list = metrics_list

        #necessary columns to construct the metrics
        join_columns = {
            "recording_filename",
            "child_id",
            "duration",
            "session_id",
            "session_offset",
        }
        #get existing columns of the dataset for recordings
        correct_cols = set(self.project.recordings.columns)
        if by not in correct_cols: exit("<{}> is not specified in this dataset, cannot extract by it, change your --by option".format(by))
        if rec_cols:
            #when user requests recording columns, build the list and verify they exist (warn otherwise)
            rec_cols=set(rec_cols.split(","))
            for i in rec_cols:
                if i not in correct_cols:
                    print("Warning, requested column <{}> does not exist in recordings.csv, ignoring this column. existing columns are : {}".format(i,correct_cols))
            rec_cols &= correct_cols
            #add wanted columns to the one we already get
            join_columns.update(rec_cols)
        self.rec_cols = rec_cols
        
        join_columns &= correct_cols
        
        #join dataset annotation with their info in recordings.csv
        self.am.annotations = self.am.annotations.merge(
            self.project.recordings[list(join_columns)],
            left_on="recording_filename",
            right_on="recording_filename",
        )
        
        #get existing columns of the dataset for children
        correct_cols = set(self.project.children.columns)
        if child_cols:
            #when user requests children columns, build the list and verify they exist (warn otherwise)
            child_cols = set(child_cols.split(","))
            child_cols.add("child_id")
            for i in child_cols:
                if i not in correct_cols:
                    print("Warning, requested column <{}> does not exist in children.csv, ignoring this column. existing columns are : {}".format(i,correct_cols))
            child_cols &= correct_cols
            self.child_cols = child_cols
    
            #join dataset annotation with their info in children.csv
            self.am.annotations = self.am.annotations.merge(
                self.project.children[list(child_cols)],
                left_on="child_id",
                right_on="child_id",
            )
        else:
            self.child_cols = None
        
        self.by = by
        self.period = period
        
        #build a dataframe with all the periods we will want for each unit
        if self.period: 
            self.periods = pd.interval_range(
                start=datetime.datetime(1900, 1, 1, 0, 0, 0, 0),
                end=datetime.datetime(1900, 1, 2, 0, 0, 0, 0),
                freq=self.period,
                closed="both",
            )
            self.periods= pd.DataFrame(self.periods.to_tuples().to_list(),columns=['period_start','period_end'])
        
        self.segments = pd.DataFrame()

        self.recordings = Pipeline.recordings_from_list(recordings)

        self.from_time = from_time
        self.to_time = to_time

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        pipelines[cls.SUBCOMMAND] = cls

    def _process_unit(self,row):
        prev_set = ""
        duration_set = 0
        for i, line in self.metrics_list.iterrows():
            curr_set = line["set"]
            if prev_set != curr_set:
                index, annotations = self.retrieve_segments([curr_set],row[1])
                duration_set = (
                        index["range_offset"] - index["range_onset"]
                    ).sum() / 1000
                row[1]["duration_{}".format(line["set"])] = duration_set
                prev_set = curr_set            
        
            if 'name' in line and not pd.isnull(line["name"]) :
                row[1][line["name"]] = eval(line["callable"])(annotations, duration_set, arguments = line["arguments"])[1]
            else :
                name, value = eval(line["callable"])(annotations, duration_set, arguments = line["arguments"])
                row[1][name] = value
                
        return row[1]
                
    
    #from the initiated self.metrics, compute each row metrics (with threads or not)
    def extract(self):
        if self.threads == 1:
            self.metrics = pd.DataFrame(
                [self._process_unit(row) for row in self.metrics.iterrows()]
            )
        else:
            with mp.Pool(
                processes=self.threads if self.threads >= 1 else mp.cpu_count()
            ) as pool:
                self.metrics = pd.DataFrame(
                    pool.map(self._process_unit, self.metrics.iterrows())
                )
            

        return self.metrics

    def retrieve_segments(self, sets: List[str], row: str):
        annotations = self.am.annotations[self.am.annotations[self.by] == row[self.by]]
        annotations = annotations[annotations["set"].isin(sets)]

        if self.from_time and self.to_time:
            annotations = self.am.get_within_time_range(
                annotations, self.from_time, self.to_time, errors="coerce"
            )
            
        if self.period:
            st_hour = datetime.datetime.strptime(row["period_start"], "%H:%M")
            end_hour = datetime.datetime.strptime(row["period_end"], "%H:%M")
            annotations = self.am.get_within_time_range(
                    annotations, st_hour, end_hour)

        try:
            segments = self.am.get_segments(annotations)
        except Exception as e:
            print(str(e))
            return pd.DataFrame(), pd.DataFrame()

        # prevent overflows
        segments["duration"] = (
            (segments["segment_offset"] / 1000 - segments["segment_onset"] / 1000)
            .astype(float)
            .fillna(0)
        )

        return annotations, segments
    
    def _initiate_metrics_df(self):
        """builds a dataframe with all the rows necessary and their labels
        eg : - one row per child if --by child_id and no --period
             - 48 rows if 2 recordings in the corpus --period 1h --by recording_filename
        Then the extract() method should populate the dataframe with actual metrics
        """
        recordings = self.project.get_recordings_from_list(self.recordings)
        self.metrics = pd.DataFrame(recordings[self.by].unique(), columns=[self.by])
        
        if self.period:
            #if period, use the self.periods dataframe to build all the list of segments per unit
            self.metrics["key"]=0 #with old versions of pandas, we are forced to have a common column to do a cross join, we drop the column after
            self.periods["key"]=0
            self.metrics = pd.merge(self.metrics,self.periods,on='key',how='outer').drop('key',axis=1)
        
        #add info for child_id and duration to the dataframe
        self.metrics["child_id"] = self.metrics.apply(
                lambda row:self.project.recordings[self.project.recordings[self.by] == row[self.by]
        ]["child_id"].iloc[0],
        axis=1)
        
        #durations will need to be computed per set because annotated time may change between sets and that will influence raw numbers and rates.
        #metrics["duration"] = unit_duration
        
        #get and add to dataframe children.csv columns asked
        if self.child_cols:
            for label in self.child_cols:
                self.metrics[label]= self.metrics.apply(lambda row:
                    self.project.children[
                        self.project.children["child_id"] == row["child_id"]
                ][label].iloc[0], axis=1)
                
        def check_unicity(row, label):
            value=self.project.recordings[
                        self.project.recordings[self.by] == row[self.by]
                ][label].drop_duplicates()
            #check that there is only one row remaining (ie this column has a unique value for that unit)
            if len(value) == 1:
                return value.iloc[0]
            #otherwise, leave the column as NA
            else:
                return "NA"
        
        #get and add to dataframe recordings.csv columns asked
        if self.rec_cols:
            for label in self.rec_cols:
                self.metrics[label] = self.metrics.apply(lambda row : check_unicity(row,label),axis=1)
        
        
class customMetrics(Metrics):
    """custom metrics extractor. 
    Extracts a number of metrics from a given list of callable functions.
    An object containing the callable functions sets desired and other arguments can be used or a path to a yaml file
    with all necessary parameters

    
    """
    
    SUBCOMMAND = "custom"

    def __init__(
        self,
        project: ChildProject.projects.ChildProject,
        parameters: Union[str, pd.DataFrame],
        recordings: Union[str, List[str], pd.DataFrame] = None,
        from_time: str = None,
        to_time: str = None,       
        rec_cols: str = None,
        child_cols: str = None,
        by: str = "recording_filename",
        period: str = None,
    ):
        
        ##TODO parse yml file to create the parameters and the metrics_list
        #probably overwrite cli arguments if they are given in the yaml file
        
        super().__init__(project, by, recordings, from_time, to_time, rec_cols, child_cols)
        
    def _process_unit(self, unit: str):
        pass
    
    def extract(self):
        pass
    
    @staticmethod
    def add_parser(subparsers, subcommand):
        parser = subparsers.add_parser(subcommand, help="custom metrics")
        parser.add_argument("parameters",
            help="name of the .yml parameter file for custom metrics",
            required=True,
        )
        parser.add_argument(
            "--threads", help="amount of threads to run on", default=1, type=int
        )
        
class LenaMetrics(Metrics):
    """LENA metrics extractor. 
    Extracts a number of metrics from the LENA .its annotations.

    :param project: ChildProject instance of the target dataset.
    :type project: ChildProject.projects.ChildProject
    :param set: name of the set associated to the .its annotations
    :type set: str
    :param recordings: recordings to sample from; if None, all recordings will be sampled, defaults to None
    :type recordings: Union[str, List[str], pd.DataFrame], optional
    :param from_time: If specified (in HH:MM format), ignore annotations outside of the given time-range, defaults to None
    :type from_time: str, optional
    :param to_time:  If specified (in HH:MM format), ignore annotations outside of the given time-range, defaults to None
    :type to_time: str, optional
    :param rec_cols: columns from recordings.csv to include in the outputted metrics (optional), recording_filename,session_id,child_id,duration are always included if possible and dont need to be specified. Any column that is not unique for a given unit (eg date_iso for a child_id being recorded on multiple days) will output a <NA> value
    :type rec_cols: str, optional
    :param child_cols: columns from children.csv to include in the outputted metrics (optional), None by default
    :type child_cols: str, optional
    :param by: units to sample from, defaults to 'recording_filename'
    :type by: str, optional
    :param threads: amount of threads to run on, defaults to 1
    :type threads: int, optional
    """

    SUBCOMMAND = "lena"

    def __init__(
        self,
        project: ChildProject.projects.ChildProject,
        set: str,
        recordings: Union[str, List[str], pd.DataFrame] = None,
        from_time: str = None,
        to_time: str = None,       
        rec_cols: str = None,
        child_cols: str = None,
        by: str = "recording_filename",
        threads: int = 1,
    ):
        self.set = set
        
        METRICS = pd.DataFrame(np.array(
            [["voc_speaker_ph",self.set,{'speaker': 'FEM'}],
             ["voc_speaker_ph",self.set,{'speaker': 'MAL'}],
             ["voc_speaker_ph",self.set,{'speaker': 'OCH'}],
             ["voc_dur_speaker_ph",self.set,{'speaker': 'FEM'}],
             ["voc_dur_speaker_ph",self.set,{'speaker': 'MAL'}],
             ["voc_dur_speaker_ph",self.set,{'speaker': 'OCH'}],
             ["avg_voc_dur_speaker",self.set,{'speaker': 'FEM'}],
             ["avg_voc_dur_speaker",self.set,{'speaker': 'MAL'}],
             ["avg_voc_dur_speaker",self.set,{'speaker': 'OCH'}],
             ["wc_speaker_ph",self.set,{'speaker': 'FEM'}],
             ["wc_speaker_ph",self.set,{'speaker': 'MAL'}],
             ["wc_adu_ph",self.set,{}],
             ["lp_n",self.set,{}],
             ["lp_dur",self.set,{}],
             ]), columns=["callable","set","arguments"])

        super().__init__(project, METRICS, by, recordings, from_time, to_time, rec_cols, child_cols, threads)

        

        if self.set not in self.am.annotations["set"].values:
            raise ValueError(
                f"annotation set '{self.set}' was not found in the index; "
                "check spelling and make sure the set was properly imported."
            )

    @staticmethod
    def add_parser(subparsers, subcommand):
        parser = subparsers.add_parser(subcommand, help="LENA metrics")
        parser.add_argument("set", help="name of the LENA its annotations set")
        parser.add_argument(
            "--threads", help="amount of threads to run on", default=1, type=int
        )


class AclewMetrics(Metrics):
    """ACLEW metrics extractor.
    Extracts a number of metrics from the ACLEW pipeline annotations, which includes:

     - The Voice Type Classifier by Lavechin et al. (arXiv:2005.12656)
     - The Automatic LInguistic Unit Count Estimator (ALICE) by Räsänen et al. (doi:10.3758/s13428-020-01460-x)
     - The VoCalisation Maturity model (VCMNet) by Al Futaisi et al. (doi:10.1145/3340555.3353751)

    :param project: ChildProject instance of the target dataset.
    :type project: ChildProject.projects.ChildProject
    :param vtc: name of the set associated to the VTC annotations
    :type vtc: str
    :param alice: name of the set associated to the ALICE annotations
    :type alice: str
    :param vcm: name of the set associated to the VCM annotations
    :type vcm: str
    :param recordings: recordings to sample from; if None, all recordings will be sampled, defaults to None
    :type recordings: Union[str, List[str], pd.DataFrame], optional
    :param from_time: If specified (in HH:MM format), ignore annotations outside of the given time-range, defaults to None
    :type from_time: str, optional
    :param to_time:  If specified (in HH:MM format), ignore annotations outside of the given time-range, defaults to None
    :type to_time: str, optional
    :param rec_cols: columns from recordings.csv to include in the outputted metrics (optional), recording_filename,session_id,child_id,duration are always included if possible and dont need to be specified. Any column that is not unique for a given unit (eg date_iso for a child_id being recorded on multiple days) will output a <NA> value
    :type rec_cols: str, optional
    :param child_cols: columns from children.csv to include in the outputted metrics (optional), None by default
    :type child_cols: str, optional
    :param by: units to sample from, defaults to 'recording_filename'
    :type by: str, optional
    :param threads: amount of threads to run on, defaults to 1
    :type threads: int, optional
    """

    SUBCOMMAND = "aclew"

    def __init__(
        self,
        project: ChildProject.projects.ChildProject,
        vtc: str = "vtc",
        alice: str = "alice",
        vcm: str = "vcm",
        recordings: Union[str, List[str], pd.DataFrame] = None,
        from_time: str = None,
        to_time: str = None,
        rec_cols: str = None,
        child_cols: str = None,
        by: str = "recording_filename",
        threads: int = 1,
    ):

        super().__init__(project, by, recordings, from_time, to_time, rec_cols, child_cols)

        self.vtc = vtc
        self.alice = alice
        self.vcm = vcm
        self.threads = int(threads)
        
        if self.vtc not in self.am.annotations["set"].values:
            raise ValueError(
                f"The VTC set '{self.vtc}' was not found in the index; "
                "check spelling and make sure the set was properly imported."
            )

        if self.alice not in self.am.annotations["set"].values:
            print(f"The ALICE set ('{self.alice}') was not found in the index.")

        if self.vcm not in self.am.annotations["set"].values:
            print(f"The VCM set ('{self.vcm}') was not found in the index.")

    def _process_unit(self, unit: str):
        metrics = {self.by: unit}
        annotations, segments = self.retrieve_segments(
            [self.vtc, self.alice, self.vcm], unit
        )

        speaker_types = ["FEM", "MAL", "CHI", "OCH"]
        adults = ["FEM", "MAL"]

        if "speaker_type" in segments.columns:
            segments = segments[segments["speaker_type"].isin(speaker_types)]
        else:
            return metrics

        if len(segments) == 0:
            return metrics

        vtc_ann = annotations[annotations["set"] == self.vtc]
        unit_duration = (vtc_ann["range_offset"] - vtc_ann["range_onset"]).sum() / 1000

        vtc = segments[segments["set"] == self.vtc]
        alice = segments[segments["set"] == self.alice]
        vcm = segments[segments["set"] == self.vcm]

        vtc_agg = vtc.groupby("speaker_type").agg(
            voc_ph=("duration", "count"),
            voc_dur_ph=("duration", "sum"),
            avg_voc_dur=("duration", "mean"),
        )

        for speaker in speaker_types:
            if speaker not in vtc_agg.index:
                continue

            metrics["voc_{}_ph".format(speaker.lower())] = (
                3600 / unit_duration
            ) * vtc_agg.loc[speaker, "voc_ph"]
            metrics["voc_dur_{}_ph".format(speaker.lower())] = (
                3600 / unit_duration
            ) * vtc_agg.loc[speaker, "voc_dur_ph"]
            metrics["avg_voc_dur_{}".format(speaker.lower())] = vtc_agg.loc[
                speaker, "avg_voc_dur"
            ]
        
        if len(alice):
            alice_agg = alice.groupby("speaker_type").agg(
                wc_ph=("words", "sum"),
                sc_ph=("syllables", "sum"),
                pc_ph=("phonemes", "sum"),
            )

            for speaker in adults:
                if speaker not in alice_agg.index:
                    continue

                metrics["wc_{}_ph".format(speaker.lower())] = (
                    3600 / unit_duration
                ) * alice_agg.loc[speaker, "wc_ph"]
                metrics["sc_{}_ph".format(speaker.lower())] = (
                    3600 / unit_duration
                ) * alice_agg.loc[speaker, "sc_ph"]
                metrics["pc_{}_ph".format(speaker.lower())] = (
                    3600 / unit_duration
                ) * alice_agg.loc[speaker, "pc_ph"]

            metrics["wc_adu_ph"] = alice["words"].sum() * 3600 / unit_duration
            metrics["sc_adu_ph"] = alice["syllables"].sum() * 3600 / unit_duration
            metrics["pc_adu_ph"] = alice["phonemes"].sum() * 3600 / unit_duration

        if len(vcm):
            vcm_agg = (
                vcm[vcm["speaker_type"] == "CHI"]
                .groupby("vcm_type")
                .agg(
                    voc_chi_ph=("duration", "count"),
                    voc_dur_chi_ph=("duration", "sum",),
                    avg_voc_dur_chi=("duration", "mean"),
                )
            )

            metrics["cry_voc_chi_ph"] = (3600 / unit_duration) * (
                vcm_agg.loc["Y", "voc_chi_ph"] if "Y" in vcm_agg.index else 0
            )
            metrics["cry_voc_dur_chi_ph"] = (3600 / unit_duration) * (
                vcm_agg.loc["Y", "voc_dur_chi_ph"] if "Y" in vcm_agg.index else 0
            )

            if "Y" in vcm_agg.index:
                metrics["avg_cry_voc_dur_chi"] = (3600 / unit_duration) * vcm_agg.loc[
                    "Y", "avg_voc_dur_chi"
                ]

            metrics["can_voc_chi_ph"] = (3600 / unit_duration) * (
                vcm_agg.loc["C", "voc_chi_ph"] if "C" in vcm_agg.index else 0
            )
            metrics["can_voc_dur_chi_ph"] = (3600 / unit_duration) * (
                vcm_agg.loc["C", "voc_dur_chi_ph"] if "C" in vcm_agg.index else 0
            )

            if "C" in vcm_agg.index:
                metrics["avg_can_voc_dur_chi"] = (3600 / unit_duration) * vcm_agg.loc[
                    "C", "avg_voc_dur_chi"
                ]

            metrics["non_can_voc_chi_ph"] = (3600 / unit_duration) * (
                vcm_agg.loc["N", "voc_chi_ph"] if "N" in vcm_agg.index else 0
            )
            metrics["non_can_voc_dur_chi_ph"] = (3600 / unit_duration) * (
                vcm_agg.loc["N", "voc_dur_chi_ph"] if "N" in vcm_agg.index else 0
            )

            if "N" in vcm_agg.index:
                metrics["avg_non_can_voc_dur_chi"] = (
                    3600 / unit_duration
                ) * vcm_agg.loc["N", "avg_voc_dur_chi"]

            speech_voc = metrics["can_voc_chi_ph"] + metrics["non_can_voc_chi_ph"]
            speech_dur = (
                metrics["can_voc_dur_chi_ph"] + metrics["non_can_voc_dur_chi_ph"]
            )

            cry_voc = metrics["cry_voc_chi_ph"]
            cry_dur = metrics["cry_voc_dur_chi_ph"]

            if speech_voc + cry_voc:
                metrics["lp_n"] = speech_voc / (speech_voc + cry_voc)
                metrics["cp_n"] = metrics["can_voc_chi_ph"] / speech_voc

                metrics["lp_dur"] = speech_dur / (speech_dur + cry_dur)
                metrics["cp_dur"] = metrics["can_voc_dur_chi_ph"] / speech_dur

        #get child_id and duration that are always given
        metrics["child_id"] = self.project.recordings[
            self.project.recordings[self.by] == unit
        ]["child_id"].iloc[0]
        metrics["duration"] = unit_duration
        
        #get and add to dataframe children.csv columns asked
        if self.child_cols:
            for label in self.child_cols:
                metrics[label]=self.project.children[
                        self.project.children["child_id"] == metrics["child_id"]
                ][label].iloc[0]
                
        #get and add to dataframe recordings.csv columns asked
        if self.rec_cols:
            for label in self.rec_cols:
                #for every unit drop the duplicates for that column
                value=self.project.recordings[
                        self.project.recordings[self.by] == unit
                ][label].drop_duplicates()
                #check that there is only one row remaining (ie this column has a unique value for that unit)
                if len(value) == 1:
                    metrics[label]=value.iloc[0]
                #otherwise, leave the column as NA
                else:
                    metrics[label]="NA"
        
        return metrics

    def extract(self):
        recordings = self.project.get_recordings_from_list(self.recordings)

        if self.threads == 1:
            self.metrics = pd.DataFrame(
                [self._process_unit(unit) for unit in recordings[self.by].unique()]
            )
        else:
            with mp.Pool(
                processes=self.threads if self.threads >= 1 else mp.cpu_count()
            ) as pool:
                self.metrics = pd.DataFrame(
                    pool.map(self._process_unit, recordings[self.by].unique())
                )

        self.metrics.set_index(self.by, inplace=True)
        return self.metrics

    @staticmethod
    def add_parser(subparsers, subcommand):
        parser = subparsers.add_parser(subcommand, help="LENA metrics")
        parser.add_argument("--vtc", help="vtc set", default="vtc")
        parser.add_argument("--alice", help="alice set", default="alice")
        parser.add_argument("--vcm", help="vcm set", default="vcm")
        parser.add_argument(
            "--threads", help="amount of threads to run on", default=1, type=int
        )


class PeriodMetrics(Metrics):
    """Time-aggregated metrics extractor.

    Aggregates vocalizations for each time-of-the-day-unit based on a period specified by the user.
    For instance, if the period is set to ``15Min`` (i.e. 15 minutes), vocalization rates will be reported for each
    recording and time-unit (e.g. 09:00 to 09:15, 09:15 to 09:30, etc.).

    The output dataframe has ``rp`` rows, where ``r`` is the amount of recordings (or children if the ``--by`` option is set to ``child_id``), ``p`` is the 
    amount of time-bins per day (i.e. 24 x 4 = 96 for a 15-minute period).

    The output dataframe includes a ``period`` column that contains the onset of each time-unit in HH:MM:SS format.
    The ``duration`` columns contains the total amount of annotations covering each time-bin, in milliseconds.

    If ``--by`` is set to e.g. ``child_id``, then the values for each time-bin will be the average rates across
    all the recordings of every child.

    :param project: ChildProject instance of the target dataset
    :type project: ChildProject.projects.ChildProject
    :param set: name of the set of annotations to derive the metrics from
    :type set: str
    :param period: Time-period. Values should be formatted as `pandas offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__. For instance, `15Min` corresponds to a 15 minute period; `2H` corresponds to a 2 hour period.
    :type period: str
    :param period_origin: NotImplemented, defaults to None
    :type period_origin: str, optional
    :param recordings: white-list of recordings to process, defaults to None
    :type recordings: Union[str, List[str], pd.DataFrame], optional
    :param from_time: If specified (in HH:MM format), ignore annotations outside of the given time-range, defaults to None
    :type from_time: str, optional
    :param to_time:  If specified (in HH:MM format), ignore annotations outside of the given time-range, defaults to None
    :type to_time: str, optional
    :param rec_cols: columns from recordings.csv to include in the outputted metrics (optional), recording_filename,session_id,child_id,duration are always included if possible and dont need to be specified. Any column that is not unique for a given unit (eg date_iso for a child_id being recorded on multiple days) will output a <NA> value
    :type rec_cols: str, optional
    :param child_cols: columns from children.csv to include in the outputted metrics (optional), None by default
    :type child_cols: str, optional
    :param by: units to sample from, defaults to 'recording_filename'
    :type by: str, optional
    :param threads: amount of threads to run on, defaults to 1
    :type threads: int, optional
    """

    SUBCOMMAND = "period"

    def __init__(
        self,
        project: ChildProject.projects.ChildProject,
        set: str,
        period: str,
        period_origin: str = None,
        recordings: Union[str, List[str], pd.DataFrame] = None,
        from_time: str = None,
        to_time: str = None,
        rec_cols: str = None,
        child_cols: str = None,
        by: str = "recording_filename",
        threads: int = 1,
    ):

        super().__init__(project, by, recordings, from_time, to_time, rec_cols, child_cols)

        self.set = set
        self.threads = int(threads)

        self.period = period
        self.period_origin = period_origin

        if self.period_origin is not None:
            raise NotImplementedError("period-origin is not supported yet")

        if self.set not in self.am.annotations["set"].values:
            raise ValueError(
                f"'{self.set}' was not found in the index; "
                "check spelling and make sure the set was properly imported."
            )

        self.periods = pd.date_range(
            start=datetime.datetime(1900, 1, 1, 0, 0, 0, 0),
            end=datetime.datetime(1900, 1, 2, 0, 0, 0, 0),
            freq=self.period,
            closed="left",
        )

    def _process_unit(self, unit: str):
        annotations, segments = self.retrieve_segments([self.set], unit)

        # retrieve timestamps for each vocalization, ignoring the day of occurence
        segments = self.am.get_segments_timestamps(segments, ignore_date=True)

        # dropping segments for which no time information is available
        segments.dropna(subset=["onset_time"], inplace=True)

        # update the timestamps so that all vocalizations appear
        # to happen on the same day
        segments["onset_time"] -= pd.to_timedelta(
            86400
            * ((segments["onset_time"] - self.periods[0]).dt.total_seconds() // 86400),
            unit="s",
        )

        if len(segments) == 0:
            return pd.DataFrame()

        # calculate length of available annotations within each bin.
        # this is necessary in order to calculate correct rates
        bins = np.array(
            [dt.total_seconds() for dt in self.periods - self.periods[0]] + [86400]
        )

        # retrieve the timestamps for all annotated portions of the recordings
        annotations = self.am.get_segments_timestamps(
            annotations, ignore_date=True, onset="range_onset", offset="range_offset"
        )

        # calculate time elapsed since the first time bin
        annotations["onset_time"] = (
            annotations["onset_time"]
            .apply(lambda dt: (dt - self.periods[0]).total_seconds())
            .astype(int)
        )
        annotations["offset_time"] = (
            annotations["offset_time"]
            .apply(lambda dt: (dt - self.periods[0]).total_seconds())
            .astype(int)
        )

        # split annotations to intervals each within a 0-24h range
        annotations["stops"] = annotations.apply(
            lambda row: [row["onset_time"]]
            + list(
                86400
                * np.arange(
                    (row["onset_time"] // 86400) + 1,
                    (row["offset_time"] // 86400) + 1,
                    1,
                )
            )
            + [row["offset_time"]],
            axis=1,
        )

        annotations = annotations.explode("stops")
        annotations["onset"] = annotations["stops"]
        annotations["offset"] = annotations["stops"].shift(-1)

        annotations.dropna(subset=["offset"], inplace=True)
        annotations["onset"] = annotations["onset"].astype(int) % 86400
        annotations["offset"] = (annotations["offset"] - 1e-4) % 86400

        durations = [
            (
                annotations["offset"].clip(bins[i], bins[i + 1])
                - annotations["onset"].clip(bins[i], bins[i + 1])
            ).sum()
            for i, t in enumerate(bins[:-1])
        ]

        durations = pd.Series(durations, index=self.periods)
        metrics = pd.DataFrame(index=self.periods)

        grouper = pd.Grouper(key="onset_time", freq=self.period, closed="left")

        speaker_types = ["FEM", "MAL", "CHI", "OCH"]
        adults = ["FEM", "MAL"]

        for speaker in speaker_types:
            vocs = segments[segments["speaker_type"] == speaker].groupby(grouper)

            vocs = vocs.agg(
                voc_ph=("segment_onset", "count"),
                voc_dur_ph=("duration", "sum"),
                avg_voc_dur=("duration", "mean"),
            )

            metrics["voc_{}_ph".format(speaker.lower())] = (
                vocs["voc_ph"].reindex(self.periods, fill_value=0) * 3600 / durations
            )
            metrics["voc_dur_{}_ph".format(speaker.lower())] = (
                vocs["voc_dur_ph"].reindex(self.periods, fill_value=0)
                * 3600
                / durations
            )
            metrics["avg_voc_dur_{}".format(speaker.lower())] = vocs[
                "avg_voc_dur"
            ].reindex(self.periods)

        #add duration and child_id to dataframe as they are always given
        metrics["duration"] = (durations * 1000).astype(int)
        metrics[self.by] = unit
        metrics["child_id"] = self.project.recordings[
            self.project.recordings[self.by] == unit
        ]["child_id"].iloc[0]
        
        #get and add to dataframe children.csv columns asked
        if self.child_cols:
            for label in self.child_cols:
                metrics[label]=self.project.children[
                        self.project.children["child_id"] == metrics["child_id"].iloc[0]
                ][label].iloc[0]
                
        #get and add to dataframe recordings.csv columns asked
        if self.rec_cols:
            for label in self.rec_cols:
                #for every unit drop the duplicates for that column
                value=self.project.recordings[
                        self.project.recordings[self.by] == unit
                ][label].drop_duplicates()
                #check that there is only one row remaining (ie this column has a unique value for that unit)
                if len(value) == 1:
                    metrics[label]=value.iloc[0]
                #otherwise, leave the column as NA
                else:
                    metrics[label]="NA"
        
        return metrics

    def extract(self):
        recordings = self.project.get_recordings_from_list(self.recordings)

        if self.threads == 1:
            self.metrics = pd.concat(
                [self._process_unit(unit) for unit in recordings[self.by].unique()]
            )
        else:
            with mp.Pool(
                processes=self.threads if self.threads >= 1 else mp.cpu_count()
            ) as pool:
                self.metrics = pd.concat(
                    pool.map(self._process_unit, recordings[self.by].unique())
                )

        if len(self.metrics):
            self.metrics["period"] = self.metrics.index.strftime("%H:%M:%S")
            self.metrics.set_index(self.by, inplace=True)

        return self.metrics

    @staticmethod
    def add_parser(subparsers, subcommand):
        parser = subparsers.add_parser(subcommand, help="LENA metrics")
        parser.add_argument("--set", help="annotations set", required=True)

        parser.add_argument(
            "--period",
            help="time units to aggregate (optional); equivalent to ``pandas.Grouper``'s freq argument.",
            required=True,
        )

        parser.add_argument(
            "--period-origin",
            help="time origin of each time period; equivalent to ``pandas.Grouper``'s origin argument.",
            default=None,
        )

        parser.add_argument(
            "--threads", help="amount of threads to run on", default=1, type=int
        )


class MetricsPipeline(Pipeline):
    def __init__(self):
        self.metrics = []

    def run(self, path, destination, pipeline, func=None, **kwargs):
        self.project = ChildProject.projects.ChildProject(path)
        self.project.read()

        if pipeline not in pipelines:
            raise NotImplementedError(f"invalid pipeline '{pipeline}'")

        metrics = pipelines[pipeline](self.project, **kwargs)
        metrics.extract()

        self.metrics = metrics.metrics
        self.metrics.to_csv(destination)

        return self.metrics

    @staticmethod
    def setup_parser(parser):
        parser.add_argument("path", help="path to the dataset")
        parser.add_argument("destination", help="segments destination")

        subparsers = parser.add_subparsers(help="pipeline", dest="pipeline")
        for pipeline in pipelines:
            pipelines[pipeline].add_parser(subparsers, pipeline)

        parser.add_argument(
            "--recordings",
            help="path to a CSV dataframe containing the list of recordings to sample from (by default, all recordings will be sampled). The CSV should have one column named recording_filename.",
            default=None,
        )

        parser.add_argument(
            "--by",
            help="units to sample from (default behavior is to sample by recording)",
            choices=["recording_filename", "session_id", "child_id","experiment"],
            default="recording_filename",
        )
        
        parser.add_argument(
            "--period",
            help="time units to aggregate (optional); equivalent to ``pandas.Grouper``'s freq argument.",
            default=None,
        )

        parser.add_argument(
            "-f",
            "--from-time",
            help="time range start in HH:MM format (optional)",
            default=None,
        )

        parser.add_argument(
            "-t",
            "--to-time",
            help="time range end in HH:MM format (optional)",
            default=None,
        )
        
        parser.add_argument(
            "--rec-cols",
            help="columns from recordings.csv to include in the outputted metrics (optional), NA if ambiguous",
            default=None,
        )
        
        parser.add_argument(
            "--rec-cols",
            help="columns from recordings.csv to include in the outputted metrics (optional), NA if ambiguous",
            nargs="+", default=[],
        )
        
        parser.add_argument(
            "--child-cols",
            help="columns from children.csv to include in the outputted metrics (optional)",
            nargs="+", default=[],
        )
        
        parser.add_argument(
            "--threads", help="amount of threads to run on", default=1, type=int
        )
