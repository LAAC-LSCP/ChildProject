from abc import ABC, abstractmethod
import os
import argparse
import datetime
import multiprocessing as mp
import numpy as np
import pandas as pd
from typing import Union, List
import yaml

import ChildProject
from ChildProject.pipelines.pipeline import Pipeline

import ChildProject.pipelines.metricsFunctions as metfunc #used by eval

pipelines = {}

class Metrics(ABC):
    def __init__(
        self,
        project: ChildProject.projects.ChildProject,
        metrics_list: pd.DataFrame,
        by: str = "recording_filename",
        recordings: Union[str, List[str], pd.DataFrame] = None,
        from_time: str = None,
        to_time: str = None,
        rec_cols: str = None,
        child_cols: str = None,
        period: str = None,
        threads: int = 1,
    ):

        self.project = project
        self.am = ChildProject.annotations.AnnotationManager(self.project)
        self.threads = int(threads)
        
        #check that the callable column is either a callable function or a string that can be evaluated as being part of the list of metrics in ChildProject/pipelines/metricsFunctions.py
        def check_callable(row):
            if callable(row["callable"]): return row["callable"]
            if isinstance(row["callable"],str):
                try:
                    f = eval("metfunc."+row["callable"])
                except Exception:
                    raise ValueError("{} function is not defined and was not found in ChildProject/pipelines/metricsFunctions.py".format(row["callable"]))
                return f
            else :
                raise ValueError("{} cannot be evaluated as a metric, must be a callable object or a string".format(row["callable"]))
        
        #block checking presence of required columns and evaluates the callable functions
        if isinstance(metrics_list, pd.DataFrame):
            if ({'callable','set'}).issubset(metrics_list.columns):
                metrics_list["callable"] = metrics_list.apply(check_callable,axis=1)
            else:
                raise ValueError("metrics_list parameter must contain atleast the columns [callable,set]")
        else:
            raise ValueError("metrics_list parameter must be a pandas DataFrame")
        metrics_list.sort_values(by="set",inplace=True)
        
        for setname in np.unique(metrics_list['set'].values):
            if setname not in self.am.annotations["set"].values:
                raise ValueError(
                    f"annotation set '{setname}' was not found in the index; "
                    "check spelling and make sure the set was properly imported."
                )
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
            self.periods = self.periods.applymap(datetime.datetime.strftime,format="%H:%M") #store as str time instead of timestamps (be careful with usage)
        
        self.segments = pd.DataFrame()

        self.recordings = Pipeline.recordings_from_list(recordings)

        self.from_time = from_time
        self.to_time = to_time
        
        self._initiate_metrics_df()

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
        
            ## TODO check that a previous metric with the same name does not already exists (if so warn/raise error)
            if 'name' in line and not pd.isnull(line["name"]) and line["name"]: #the 'name' column exists and its value is not NaN or ''  => use the name given by the user
                row[1][line["name"]] = line["callable"](annotations, duration_set, **line.drop(['callable', 'set','name'],errors='ignore').dropna().to_dict())[1]
            else : # use the default name of the metric function
                name, value = line["callable"](annotations, duration_set, **line.drop(['callable', 'set','name'],errors='ignore').dropna().to_dict())
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
            st_hour = row["period_start"]
            end_hour = row["period_end"]
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
                
class CsvMetrics(Metrics):
    """metrics extraction from a csv file. 
    Extracts a number of metrics listed in a csv file as s dataframe.
    the csv file must contain the columns :
        - 'callable' which is the name of the wanted metric from the list of available metrics
        - 'set' which is the set of annotations to use for that specific metric (make sur this set has the required columns for that metric)
        - 'name' is optional, this is the name to give to that metric (if not given, a default name will be attributed)
        - any other necessary argument for the given metrics (eg the voc_speaker_ph metric requires the 'speaker' argument: add a column 'speaker' in the csv file and fill its cells for this metric with the wanted value (CHI|FEM|MAL|OCH))
    """
    
    SUBCOMMAND = "csv"

    def __init__(
        self,
        project: ChildProject.projects.ChildProject,
        metrics: str,
        recordings: Union[str, List[str], pd.DataFrame] = None,
        from_time: str = None,
        to_time: str = None,       
        rec_cols: str = None,
        child_cols: str = None,
        by: str = "recording_filename",
        period: str = None,
        threads: int = 1,
    ):
        
        metrics_df = pd.read_csv(metrics)
        
        super().__init__(project, metrics_df, by=by, recordings=recordings, from_time=from_time, to_time=to_time, rec_cols=rec_cols, child_cols=child_cols, period=period, threads=threads)
    
    @staticmethod
    def add_parser(subparsers, subcommand):
        parser = subparsers.add_parser(subcommand, help="metrics from a csv file")
        parser.add_argument("metrics",
            help="name if the csv file containing the list of metrics",
        )
        
        
class CustomMetrics(Metrics):
    """custom metrics extractor. 
    Extracts a number of metrics from a given list of callable functions.
    An object containing the callable functions sets desired and other arguments can be used or a path to a yaml file
    with all necessary parameters

    
    """
    
    SUBCOMMAND = "custom"

    def __init__(
        self,
        project: ChildProject.projects.ChildProject,
        parameters: str,
        recordings: Union[str, List[str], pd.DataFrame] = None,
        from_time: str = None,
        to_time: str = None,       
        rec_cols: str = None,
        child_cols: str = None,
        by: str = "recording_filename",
        period: str = None,
        threads: int = 1,
    ):
        
        ##TODO parse yml file to create the parameters and the metrics_list
        #probably overwrite cli arguments if they are given in the yaml file
        params = None
        with open(parameters, "r") as stream:
            try:
                params = yaml.safe_load(stream)
                #print(params)
            except yaml.YAMLError as exc:
                raise yaml.YAMLError("parsing of the parameters file {} failed. See above exception for more details".format(parameters)) from exc
                
        if params:
            if "metrics" not in params : raise ValueError("the parameter file {} must contain at least the 'metrics' key containing the list of the desired metrics".format(parameters))
            try:
                metrics_df = pd.DataFrame(params["metrics"])
            except Exception as e:
                raise ValueError("The 'metrics' key in {} must be a list of elements".format(parameters)) from e           
        else:
            raise ValueError("could not find any parameters in {}".format(parameters))
            
        if 'recordings' in params: recordings = params['recordings'] 
        if 'from-time' in params: from_time = params['from-time'] 
        if 'to-time' in params: to_time = params['to-time'] 
        if 'rec-cols' in params: rec_cols = params['rec-cols'] 
        if 'child-cols' in params: child_cols = params['child-cols'] 
        if 'by' in params: by = params['by'] 
        if 'period' in params: period = params['period']
        if 'threads' in params: threads = params['threads'] 
        
        super().__init__(project, metrics_df, by=by, recordings=recordings, from_time=from_time, to_time=to_time, rec_cols=rec_cols, child_cols=child_cols, period=period, threads=threads)
    
    @staticmethod
    def add_parser(subparsers, subcommand):
        parser = subparsers.add_parser(subcommand, help="custom metrics")
        parser.add_argument("parameters",
            help="name of the .yml parameter file for custom metrics",
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
             ["voc_speaker_ph",self.set,{'speaker': 'CHI'}],
             ["voc_dur_speaker_ph",self.set,{'speaker': 'FEM'}],
             ["voc_dur_speaker_ph",self.set,{'speaker': 'MAL'}],
             ["voc_dur_speaker_ph",self.set,{'speaker': 'OCH'}],
             ["voc_dur_speaker_ph",self.set,{'speaker': 'CHI'}],
             ["avg_voc_dur_speaker",self.set,{'speaker': 'FEM'}],
             ["avg_voc_dur_speaker",self.set,{'speaker': 'MAL'}],
             ["avg_voc_dur_speaker",self.set,{'speaker': 'OCH'}],
             ["avg_voc_dur_speaker",self.set,{'speaker': 'CHI'}],
             ["wc_speaker_ph",self.set,{'speaker': 'FEM'}],
             ["wc_speaker_ph",self.set,{'speaker': 'MAL'}],
             ["wc_adu_ph",self.set,{}],
             ["lp_n",self.set,{}],
             ["lp_dur",self.set,{}],
             ]), columns=["callable","set","arguments"])

        super().__init__(project, METRICS, by=by, recordings=recordings, from_time=from_time, to_time=to_time, rec_cols=rec_cols, child_cols=child_cols, threads=threads)

        

        if self.set not in self.am.annotations["set"].values:
            raise ValueError(
                f"annotation set '{self.set}' was not found in the index; "
                "check spelling and make sure the set was properly imported."
            )

    @staticmethod
    def add_parser(subparsers, subcommand):
        parser = subparsers.add_parser(subcommand, help="LENA metrics")
        parser.add_argument("set", help="name of the LENA its annotations set")

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
        
        self.vtc = vtc
        self.alice = alice
        self.vcm = vcm
        
        am = ChildProject.annotations.AnnotationManager(project) #temporary instance to check for existing sets. This is suboptimal because an annotation manager will be created by Metrics. However, the metrics classe raises a ValueError for every set passed that does not exist, here we want to check in advance which of the alice and vcm sets exist without raising an error
              
        METRICS = np.array(
            [["voc_speaker_ph",self.vtc,{'speaker': 'FEM'}],
             ["voc_speaker_ph",self.vtc,{'speaker': 'MAL'}],
             ["voc_speaker_ph",self.vtc,{'speaker': 'OCH'}],
             ["voc_speaker_ph",self.vtc,{'speaker': 'CHI'}],
             ["voc_dur_speaker_ph",self.vtc,{'speaker': 'FEM'}],
             ["voc_dur_speaker_ph",self.vtc,{'speaker': 'MAL'}],
             ["voc_dur_speaker_ph",self.vtc,{'speaker': 'OCH'}],
             ["voc_dur_speaker_ph",self.vtc,{'speaker': 'CHI'}],
             ["avg_voc_dur_speaker",self.vtc,{'speaker': 'FEM'}],
             ["avg_voc_dur_speaker",self.vtc,{'speaker': 'MAL'}],
             ["avg_voc_dur_speaker",self.vtc,{'speaker': 'OCH'}],
             ["avg_voc_dur_speaker",self.vtc,{'speaker': 'CHI'}],
             ])
        
        if self.alice not in am.annotations["set"].values:
            print(f"The ALICE set ('{self.alice}') was not found in the index.")
        else:
            METRICS = np.append(METRICS,np.array(
             [["wc_speaker_ph",self.alice,{'speaker': 'FEM'}],
             ["wc_speaker_ph",self.alice,{'speaker': 'MAL'}],
             ["sc_speaker_ph",self.alice,{'speaker': 'FEM'}],
             ["sc_speaker_ph",self.alice,{'speaker': 'MAL'}],
             ["pc_speaker_ph",self.alice,{'speaker': 'FEM'}],
             ["pc_speaker_ph",self.alice,{'speaker': 'MAL'}],
             ["wc_adu_ph",self.alice,{}],
             ["sc_adu_ph",self.alice,{}],
             ["pc_adu_ph",self.alice,{}],
             ]))
             
        if self.vcm not in am.annotations["set"].values:
            print(f"The vcm set ('{self.vcm}') was not found in the index.")
        else:
            METRICS = np.append(METRICS,np.array(
             [["cry_voc_chi_ph",self.vcm,{}],
             ["cry_voc_dur_chi_ph",self.vcm,{}],
             ["avg_cry_voc_dur_chi",self.vcm,{}],
             ["can_voc_chi_ph",self.vcm,{}],
             ["can_voc_dur_chi_ph",self.vcm,{}],
             ["avg_can_voc_dur_chi",self.vcm,{}],
             ["non_can_voc_chi_ph",self.vcm,{}],
             ["non_can_voc_dur_chi_ph",self.vcm,{}],
             ["avg_non_can_voc_dur_chi",self.vcm,{}],
             ["lp_n",self.vcm,{}],
             ["lp_dur",self.vcm,{}],
             ["cp_n",self.vcm,{}],
             ["cp_dur",self.vcm,{}],
             ]))
                
        METRICS = pd.DataFrame(METRICS, columns=["callable","set","arguments"])

        super().__init__(project, METRICS,by=by, recordings=recordings, from_time=from_time, to_time=to_time, rec_cols=rec_cols, child_cols=child_cols, threads=threads)

    @staticmethod
    def add_parser(subparsers, subcommand):
        parser = subparsers.add_parser(subcommand, help="LENA metrics")
        parser.add_argument("--vtc", help="vtc set", default="vtc")
        parser.add_argument("--alice", help="alice set", default="alice")
        parser.add_argument("--vcm", help="vcm set", default="vcm")


class MetricsPipeline(Pipeline):
    def __init__(self):
        self.metrics = []

    def run(self, path, destination, pipeline, func=None, **kwargs):
        #build a dictionary with all parameters used
        parameters = locals()
        parameters = {
            key: parameters[key]
            for key in parameters
            if key not in ["self", "kwargs", "func"] #not sure what func parameter is for, seems unecessary to keep
        }
        for key in kwargs: #add all kwargs to dictionary
            parameters[key] = kwargs[key]
        
        self.project = ChildProject.projects.ChildProject(path)
        self.project.read()

        if pipeline not in pipelines:
            raise NotImplementedError(f"invalid pipeline '{pipeline}'")

        metrics = pipelines[pipeline](self.project, **kwargs)
        metrics.extract()

        self.metrics = metrics.metrics
        self.metrics.to_csv(destination)
        
        # get the df of metrics used from the Metrics class
        metrics_df = metrics.metrics_list
        metrics_df['callable'] = metrics_df.apply(lambda row: row['callable'].__name__, axis=1) #from the callables used, find their name back
        parameters['metrics_list'] = metrics_df.to_dict(orient='records')
        date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # create a yaml file with all the parameters used
        parameters_path = os.path.splitext(destination)[0] + "_parameters.yml"
        print("exported metrics to {}".format(destination))
        yaml.dump(
            {
                "package_version": ChildProject.__version__,
                "date": date,
                "parameters": parameters,
            },
            open(parameters_path, "w+"),sort_keys=False,
        )
        print("exported sampler parameters to {}".format(parameters_path))

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
            "--child-cols",
            help="columns from children.csv to include in the outputted metrics (optional), NA if ambiguous",
            default=None,
        )
        
        
        parser.add_argument(
            "--threads", help="amount of threads to run on", default=1, type=int
        )
