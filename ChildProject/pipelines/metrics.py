from abc import ABC, abstractmethod
import os
import argparse
import datetime
import multiprocessing as mp
import numpy as np
import pandas as pd
from typing import Union, List
import yaml
from git import Repo
from git.exc import InvalidGitRepositoryError

from ..projects import ChildProject
from ..annotations import AnnotationManager
from ..tables import assert_dataframe, assert_columns_presence, read_csv_with_dtype
from ..utils import TimeInterval, time_intervals_intersect
from .pipeline import Pipeline
from . import metricsFunctions as metfunc

from ChildProject import __version__

pipelines = {}

class Metrics(ABC):
    """
    Main class for generating metrics from a project object and a list of desired metrics
    
    :param project: ChildProject instance of the target dataset.
    :type project: ChildProject.projects.ChildProject
    :param metrics_list: pandas DataFrame containing the desired metrics (metrics functions are in metricsFunctions.py)
    :type metrics_list: pd.DataFrame
    :param by: unit to extract metric from (recording_filename, experiment, child_id, session_id, segments), defaults to 'recording_filename', 'segments' is mandatory if passing the segments argument
    :type by: str, optional
    :param recordings: recordings to sample from; if None, all recordings will be sampled, defaults to None
    :type recordings: Union[str, List[str], pd.DataFrame], optional
    :param from_time: If specified (in HH:MM:SS format), ignore annotations outside of the given time-range, defaults to None
    :type from_time: str, optional
    :param to_time:  If specified (in HH:MM:SS format), ignore annotations outside of the given time-range, defaults to None
    :type to_time: str, optional
    :param rec_cols: comma separated columns from recordings.csv to include in the outputted metrics (optional), recording_filename,session_id,child_id,duration are always included if possible and dont need to be specified. Any column that is not unique for a given unit (eg date_iso for a child_id being recorded on multiple days) will output a <NA> value
    :type rec_cols: str, optional
    :param child_cols: comma separated columns from children.csv to include in the outputted metrics (optional), None by default
    :type child_cols: str, optional
    :param period: time units to aggregate (optional); equivalent to ``pandas.Grouper`` freq argument.
    :type period: str, optional
    :param segments: DataFrame or path to csv file of the segments to extract from, containing 'recording_filename', 'segment_onset' and 'segment_offset' columns. To use this option, the option must be set to 'segments'. Also, this option cannot be combined with options [recordings,period,from_time,to_time].
    :type segments: Union[str, pd.DataFrame], optional
    :param threads: amount of threads to run on, defaults to 1
    :type threads: int, optional
    """
    def __init__(
        self,
        project: ChildProject,
        metrics_list: pd.DataFrame,
        by: str = "recording_filename",
        recordings: Union[str, List[str], pd.DataFrame] = None,
        from_time: str = None,
        to_time: str = None,
        rec_cols: str = None,
        child_cols: str = None,
        period: str = None,
        segments: Union[str, pd.DataFrame] = None,
        threads: int = 1,
    ):

        self.project = project
        self.am = AnnotationManager(self.project)
        self.threads = int(threads)
        
        #check that the callable column is either a callable function or a string that can be found as being part of the list of metrics in ChildProject/pipelines/metricsFunctions.py
        def check_callable(row):
            if callable(row["callable"]): return row["callable"]
            if isinstance(row["callable"],str):
                try:
                    f = getattr(metfunc , row["callable"])
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
        if by != 'segments' and by not in correct_cols: raise ValueError("<{}> is not specified in this dataset, cannot extract by it, change your --by option".format(by))
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
        self.segments = segments   
        self.recordings = Pipeline.recordings_from_list(recordings)
        
        #If the extraction is done on segments
        if segments is not None:
            #we enforce that incompatible arguments are not set
            assert by == 'segments' and period is None and recordings is None and from_time is None and to_time is None, "the <segments> option can not be combined with options [period,recordings,from_time,to_time], and <by> should be set to 'segments'"
            
            dtypes = {'recording_filename':'string','segment_onset':'Int64', 'segment_offset':'Int64'}
            #use the DataFrame provided or import it from a csv file
            if isinstance(segments, pd.DataFrame):
                self.segments = segments.astype(dtypes)
            else:
                self.segments = read_csv_with_dtype(segments,dtypes)

            #check that required columns are present and dataframe not empty
            assert_dataframe("segments", self.segments, not_empty=True)
            assert_columns_presence(
                "segments",
                self.segments,
                {"recording_filename", "segment_onset", "segment_offset"},
            )
        #not on segments
        else:
            
            #build a dataframe with all the periods we will want for each unit
            if self.period: 
                self.periods = pd.interval_range(
                    start=datetime.datetime(1900, 1, 1, 0, 0, 0, 0),
                    end=datetime.datetime(1900, 1, 2, 0, 0, 0, 0),
                    freq=self.period,
                    closed="both",
                )
                self.periods= pd.DataFrame(self.periods.to_tuples().to_list(),columns=['period_start','period_end'])
    
            #turn from_time and to to_time to datetime objects
            if from_time:
                try:
                    self.from_time = datetime.datetime.strptime(from_time, "%H:%M:%S")
                except:
                    raise ValueError(f"invalid value for from_time ('{from_time}'); should have HH:MM:SS format instead")
            else:
                self.from_time = None
            
            if to_time:
                try:
                    self.to_time = datetime.datetime.strptime(to_time, "%H:%M:%S")
                except:
                    raise ValueError(f"invalid value for to_time ('{to_time}'); should have HH:MM:SS format instead")
            else:
                self.to_time = None
        
        self._initiate_metrics_df()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        pipelines[cls.SUBCOMMAND] = cls

    def _process_unit(self,row):
        """for one unit (i.e. 1 {recording|session|child} [period]) compute the list of required metrics and store the results in the current row of self.metrics
        
        :param row: index and Series of the unit to process, to be modified with the results
        :type row: (int , pandas.Series)
        :return: Series containing all the computed metrics result for that unit
        :rtype: pandas.Series
        """
        #row[0] is the index of the row we are processing
        #row[1] is the actual Series containing all the metrics for the currently processed line
        prev_set = ""
        duration_set = 0
        for i, line in self.metrics_list.iterrows():
            curr_set = line["set"]
            if prev_set != curr_set:
                index, annotations = self.retrieve_segments([curr_set],row[1])
                if index.shape[0]:
                    duration_set = (
                            index["range_offset"] - index["range_onset"]
                        ).sum()
                else: duration_set = 0
                row[1]["duration_{}".format(line["set"])] = duration_set
                prev_set = curr_set            

            name, value = line["callable"](annotations, duration_set, **line.drop(['callable', 'set']).dropna().to_dict())
            row[1][name] = value
        
        return row[1]
                
    
    def extract(self):
        """from the initiated self.metrics, compute each row metrics (handles threading)
        Once the Metrics class is initialized, call this function to extract the metrics and populate self.metrics
        
        :return: DataFrame of computed metrics
        :rtype: pandas.DataFrame
        """
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
        if self.period:
            self.metrics['period_start'] = self.metrics['period_start'].dt.strftime('%H:%M:%S')
            self.metrics['period_end'] = self.metrics['period_end'].dt.strftime('%H:%M:%S')
        return self.metrics

    def retrieve_segments(self, sets: List[str], row: str):
        """from a list of sets and a row identifying the unit computed, return the relevant annotation segments
        
        :param sets: List of annotation sets to keep
        :type sets: List[str]
        :param row: Series storing the unit to compute information
        :type row: pandas.Series
        :return: relevant annotation DataFrame and index DataFrame
        :rtype: (pandas.DataFrame , pandas.DataFrame)
        """
        #if extraction from segments, annotations are retrieved from get_within_ranges
        if self.segments is not None:
            matches = self.am.get_within_ranges(ranges= pd.DataFrame(
                    [[row['recording_filename'],row['segment_onset'], row['segment_offset']]],
                    columns=['recording_filename','range_onset', 'range_offset']),
                    sets= sets,
                    missing_data = 'warn')
        # else prepare and use get_within_time_range
        else:
            annotations = self.am.annotations[self.am.annotations[self.by] == row[self.by]]
            annotations = annotations[annotations["set"].isin(sets)]
            # restrict to time ranges
            if self.from_time and self.to_time:
                # add the periods columns
                if self.period:
                    st_hour = row["period_start"]
                    end_hour = row["period_end"]
                    intervals = time_intervals_intersect(TimeInterval(self.from_time,self.to_time),TimeInterval(st_hour,end_hour))
                    matches = pd.concat([self.am.get_within_time_range(annotations,i) for i in intervals],ignore_index =True) if intervals else pd.DataFrame()
                else:
                    matches = self.am.get_within_time_range(
                    annotations, TimeInterval(self.from_time, self.to_time))
            elif self.period:
                # add the periods columns
                st_hour = row["period_start"]
                end_hour = row["period_end"]
                matches = self.am.get_within_time_range(
                        annotations, TimeInterval(st_hour,end_hour))
            else:
                matches = annotations

        if matches.shape[0]:
            segments = self.am.get_segments(matches)
        else:
            # no annotations for that unit
            return pd.DataFrame(), pd.DataFrame()

        # prevent overflows
        segments["duration"] = (
            (segments["segment_offset"] - segments["segment_onset"])
            .astype(float)
            .fillna(0)
        )

        return matches, segments
    
    def _initiate_metrics_df(self):
        """builds a dataframe with all the rows necessary and their labels
        eg : - one row per child if --by child_id and no --period
             - 48 rows if 2 recordings in the corpus --period 1h --by recording_filename
        Then the extract() method should populate the dataframe with actual metrics
        """
        #build the metrics dataframe from the segments argument
        if self.segments is not None:
            recordings = self.project.get_recordings_from_list(self.segments['recording_filename'].unique())
            self.by = 'recording_filename'
            self.metrics = self.segments.copy()
        # else use the list of recordings of the dataset and the by option
        else:
            recordings = self.project.get_recordings_from_list(self.recordings)
            self.metrics = pd.DataFrame(recordings[self.by].unique(), columns=[self.by])
            if self.period:
                #if period, use the self.periods dataframe to build all the list of segments per unit
                self.metrics["key"]=0 #with old versions of pandas, we are forced to have a common column to do a cross join, we drop the column after
                self.periods["key"]=0
                self.metrics = pd.merge(self.metrics,self.periods,on='key',how='outer').drop('key',axis=1)
        
        # add info for child_id
        self.metrics["child_id"] = self.metrics.apply(
                lambda row:self.project.recordings[self.project.recordings[self.by] == row[self.by]
        ]["child_id"].iloc[0],
        axis=1)
        
        # get and add to dataframe children.csv columns asked
        if self.child_cols:
            for label in self.child_cols:
                self.metrics[label]= self.metrics.apply(lambda row:
                    self.project.children[
                        self.project.children["child_id"] == row["child_id"]
                ][label].iloc[0], axis=1)
            
        # this loop is for the purpose of checking for name duplicates in the metrics
        # we do a dry run on the first line with no annotations bc impractical to check in multiprocessing
        df = pd.DataFrame()
        duration_set = 0
        names = set()
        for i, line in self.metrics_list.iterrows():
            name, value = line["callable"](df, duration_set, **line.drop(['callable', 'set'],
                                                                         errors='ignore').dropna().to_dict())
                
            if name in names:
                raise ValueError('the metric name <{}> is used multiple times, make sure it is unique'.format(name))
            else:
                names.add(name)
            
        # checking that columns added by the user are unique (e.g. date_iso may be different when extract by child_id), replace with NA if they are not
        def check_unicity(row, label):
            value=self.project.recordings[
                        self.project.recordings[self.by] == row[self.by]
                ][label].drop_duplicates()
            # check that there is only one row remaining (ie this column has a unique value for that unit)
            if len(value) == 1:
                return value.iloc[0]
            # otherwise, leave the column as NA
            else:
                return np.nan
        
        #get and add to dataframe recordings.csv columns asked
        if self.rec_cols:
            for label in self.rec_cols:
                self.metrics[label] = self.metrics.apply(lambda row : check_unicity(row,label),axis=1)
                
class CustomMetrics(Metrics):
    """metrics extraction from a csv file. 
    Extracts a number of metrics listed in a csv file as a dataframe.
    the csv file must contain the columns :
    - 'callable' which is the name of the wanted metric from the list of available metrics
    - 'set' which is the set of annotations to use for that specific metric (make sure this set has the required columns for that metric)
    - 'name' is optional, this is the name to give to that metric (if not given, a default name will be attributed)
    - any other necessary argument for the given metrics (eg the voc_speaker_ph metric requires the 'speaker' argument: add a column 'speaker' in the csv file and fill its cells for this metric with the wanted value (CHI|FEM|MAL|OCH))
        
    :param project: ChildProject instance of the target dataset.
    :type project: ChildProject.projects.ChildProject
    :param metrics: name of the csv file listing the metrics to extract
    :type metrics: str
    :param recordings: recordings to sample from; if None, all recordings will be sampled, defaults to None
    :type recordings: Union[str, List[str], pd.DataFrame], optional
    :param from_time: If specified (in HH:MM:SS format), ignore annotations outside of the given time-range, defaults to None
    :type from_time: str, optional
    :param to_time:  If specified (in HH:MM:SS format), ignore annotations outside of the given time-range, defaults to None
    :type to_time: str, optional
    :param rec_cols: comma separated columns from recordings.csv to include in the outputted metrics (optional), recording_filename,session_id,child_id,duration are always included if possible and dont need to be specified. Any column that is not unique for a given unit (eg date_iso for a child_id being recorded on multiple days) will output a <NA> value
    :type rec_cols: str, optional
    :param child_cols: comma separated columns from children.csv to include in the outputted metrics (optional), None by default
    :type child_cols: str, optional
    :param by: unit to extract metric from (recording_filename, experiment, child_id, session_id, segments), defaults to 'recording_filename', 'segments' is mandatory if passing the segments argument
    :type by: str, optional
    :param period: time units to aggregate (optional); equivalent to ``pandas.Grouper`` freq argument.
    :type period: str, optional
    :param segments: DataFrame or path to csv file of the segments to extract from, containing 'recording_filename', 'segment_onset' and 'segment_offset' columns. To use this option, the option must be set to 'segments'. Also, this option cannot be combined with options [recordings,period,from_time,to_time].
    :type segments: Union[str, pd.DataFrame], optional
    :param threads: amount of threads to run on, defaults to 1
    :type threads: int, optional
    """
    
    SUBCOMMAND = "custom"

    def __init__(
        self,
        project: ChildProject,
        metrics: str,
        recordings: Union[str, List[str], pd.DataFrame] = None,
        from_time: str = None,
        to_time: str = None,       
        rec_cols: str = None,
        child_cols: str = None,
        by: str = "recording_filename",
        period: str = None,
        segments: Union[str, pd.DataFrame] = None,
        threads: int = 1,
    ):
        
        metrics_df = pd.read_csv(metrics)
        
        super().__init__(project, metrics_df, by=by, recordings=recordings,
             from_time=from_time, to_time=to_time, rec_cols=rec_cols,
             child_cols=child_cols, period=period, segments=segments, threads=threads)
    
    @staticmethod
    def add_parser(subparsers, subcommand):
        parser = subparsers.add_parser(subcommand, help="metrics from a csv file")
        parser.add_argument("metrics",
            help="name if the csv file containing the list of metrics",
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
    :param from_time: If specified (in HH:MM:SS format), ignore annotations outside of the given time-range, defaults to None
    :type from_time: str, optional
    :param to_time:  If specified (in HH:MM:SS format), ignore annotations outside of the given time-range, defaults to None
    :type to_time: str, optional
    :param rec_cols: comma separated columns from recordings.csv to include in the outputted metrics (optional), recording_filename,session_id,child_id,duration are always included if possible and dont need to be specified. Any column that is not unique for a given unit (eg date_iso for a child_id being recorded on multiple days) will output a <NA> value
    :type rec_cols: str, optional
    :param child_cols: comma separated columns from children.csv to include in the outputted metrics (optional), None by default
    :type child_cols: str, optional
    :param by: unit to extract metric from (recording_filename, experiment, child_id, session_id, segments), defaults to 'recording_filename', 'segments' is mandatory if passing the segments argument
    :type by: str, optional
    :param period: time units to aggregate (optional); equivalent to ``pandas.Grouper`` freq argument.
    :type period: str, optional
    :param segments: DataFrame or path to csv file of the segments to extract from, containing 'recording_filename', 'segment_onset' and 'segment_offset' columns. To use this option, the option must be set to 'segments'. Also, this option cannot be combined with options [recordings,period,from_time,to_time].
    :type segments: Union[str, pd.DataFrame], optional
    :param threads: amount of threads to run on, defaults to 1
    :type threads: int, optional
    """

    SUBCOMMAND = "lena"

    def __init__(
        self,
        project: ChildProject,
        set: str,
        recordings: Union[str, List[str], pd.DataFrame] = None,
        from_time: str = None,
        to_time: str = None,       
        rec_cols: str = None,
        child_cols: str = None,
        by: str = "recording_filename",
        period: str = None,
        segments: Union[str, pd.DataFrame] = None,
        threads: int = 1,
    ):
        self.set = set
        
        METRICS = pd.DataFrame(np.array(
            [["voc_speaker_ph",self.set,'FEM'],
             ["voc_speaker_ph",self.set,'MAL'],
             ["voc_speaker_ph",self.set,'OCH'],           
             ["voc_speaker_ph",self.set,'CHI'],
             ["voc_dur_speaker_ph",self.set,'FEM'],
             ["voc_dur_speaker_ph",self.set,'MAL'],
             ["voc_dur_speaker_ph",self.set,'OCH'],
             ["voc_dur_speaker_ph",self.set,'CHI'],
             ["avg_voc_dur_speaker",self.set,'FEM'],
             ["avg_voc_dur_speaker",self.set,'MAL'],
             ["avg_voc_dur_speaker",self.set,'OCH'],
             ["avg_voc_dur_speaker",self.set,'CHI'],
             ["wc_speaker_ph",self.set,'FEM'],
             ["wc_speaker_ph",self.set,'MAL'],
             ["wc_adu_ph",self.set,pd.NA],
             ["lp_n",self.set,pd.NA],
             ["lp_dur",self.set,pd.NA],
             ["lena_CVC",self.set,pd.NA],
             ["lena_CTC",self.set,pd.NA],
             ]), columns=["callable","set","speaker"])

        super().__init__(project, METRICS, by=by, recordings=recordings,
             period=period, from_time=from_time, to_time=to_time, rec_cols=rec_cols,
             child_cols=child_cols, segments=segments, threads=threads)

        

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
    :param from_time: If specified (in HH:MM:SS format), ignore annotations outside of the given time-range, defaults to None
    :type from_time: str, optional
    :param to_time:  If specified (in HH:MM:SS format), ignore annotations outside of the given time-range, defaults to None
    :type to_time: str, optional
    :param rec_cols: comma separated columns from recordings.csv to include in the outputted metrics (optional), recording_filename,session_id,child_id,duration are always included if possible and dont need to be specified. Any column that is not unique for a given unit (eg date_iso for a child_id being recorded on multiple days) will output a <NA> value
    :type rec_cols: str, optional
    :param child_cols: comma separated columns from children.csv to include in the outputted metrics (optional), None by default
    :type child_cols: str, optional
    :param by: unit to extract metric from (recording_filename, experiment, child_id, session_id, segments), defaults to 'recording_filename', 'segments' is mandatory if passing the segments argument
    :type by: str, optional
    :param period: time units to aggregate (optional); equivalent to ``pandas.Grouper`` freq argument.
    :type period: str, optional
    :param segments: DataFrame or path to csv file of the segments to extract from, containing 'recording_filename', 'segment_onset' and 'segment_offset' columns. To use this option, the option must be set to 'segments'. Also, this option cannot be combined with options [recordings,period,from_time,to_time].
    :type segments: Union[str, pd.DataFrame], optional
    :param threads: amount of threads to run on, defaults to 1
    :type threads: int, optional
    """

    SUBCOMMAND = "aclew"

    def __init__(
        self,
        project: ChildProject,
        vtc: str = "vtc",
        alice: str = "alice",
        vcm: str = "vcm",
        recordings: Union[str, List[str], pd.DataFrame] = None,
        from_time: str = None,
        to_time: str = None,
        rec_cols: str = None,
        child_cols: str = None,
        period: str = None,
        segments: Union[str, pd.DataFrame] = None,
        by: str = "recording_filename",
        threads: int = 1,
    ):
        
        self.vtc = vtc
        self.alice = alice
        self.vcm = vcm
        
        am = AnnotationManager(project) #temporary instance to check for existing sets. This is suboptimal because an annotation manager will be created by Metrics. However, the metrics class raises a ValueError for every set passed that does not exist, here we want to check in advance which of the alice and vcm sets exist without raising an error
              
        METRICS = np.array(
            [["voc_speaker_ph",self.vtc,'FEM'],
             ["voc_speaker_ph",self.vtc,'MAL'],
             ["voc_speaker_ph",self.vtc,'OCH'],
             ["voc_speaker_ph",self.vtc,'CHI'],
             ["voc_dur_speaker_ph",self.vtc,'FEM'],
             ["voc_dur_speaker_ph",self.vtc,'MAL'],
             ["voc_dur_speaker_ph",self.vtc,'OCH'],
             ["voc_dur_speaker_ph",self.vtc,'CHI'],
             ["avg_voc_dur_speaker",self.vtc,'FEM'],
             ["avg_voc_dur_speaker",self.vtc,'MAL'],
             ["avg_voc_dur_speaker",self.vtc,'OCH'],
             ["avg_voc_dur_speaker",self.vtc,'CHI'],
             ["simple_CTC_ph",self.vtc,pd.NA],
             ])
        
        if self.alice not in am.annotations["set"].values:
            print(f"The ALICE set ('{self.alice}') was not found in the index.")
        else:
            METRICS = np.concatenate((METRICS,np.array(
             [["wc_speaker_ph",self.alice,'FEM'],
             ["wc_speaker_ph",self.alice,'MAL'],
             ["sc_speaker_ph",self.alice,'FEM'],
             ["sc_speaker_ph",self.alice,'MAL'],
             ["pc_speaker_ph",self.alice,'FEM'],
             ["pc_speaker_ph",self.alice,'MAL'],
             ["wc_adu_ph",self.alice,pd.NA],
             ["sc_adu_ph",self.alice,pd.NA],
             ["pc_adu_ph",self.alice,pd.NA],
             ])))
             
        if self.vcm not in am.annotations["set"].values:
            print(f"The vcm set ('{self.vcm}') was not found in the index.")
        else:
            METRICS = np.concatenate((METRICS,np.array(
             [["cry_voc_speaker_ph",self.vcm,'CHI'],
             ["cry_voc_dur_speaker_ph",self.vcm,'CHI'],
             ["avg_cry_voc_dur_speaker",self.vcm,'CHI'],
             ["can_voc_speaker_ph",self.vcm,'CHI'],
             ["can_voc_dur_speaker_ph",self.vcm,'CHI'],
             ["avg_can_voc_dur_speaker",self.vcm,'CHI'],
             ["non_can_voc_speaker_ph",self.vcm,'CHI'],
             ["non_can_voc_dur_speaker_ph",self.vcm,'CHI'],
             ["avg_non_can_voc_dur_speaker",self.vcm,'CHI'],
             ["lp_n",self.vcm,pd.NA],
             ["lp_dur",self.vcm,pd.NA],
             ["cp_n",self.vcm,pd.NA],
             ["cp_dur",self.vcm,pd.NA],
             ])))
                
        METRICS = pd.DataFrame(METRICS, columns=["callable","set","speaker"])

        super().__init__(project, METRICS,by=by, recordings=recordings,
             period=period, from_time=from_time, to_time=to_time,
             rec_cols=rec_cols, child_cols=child_cols, segments=segments,
             threads=threads)

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
        self.destination = destination
        #build a dictionary with all parameters used
        parameters = locals()
        parameters = {
            key: parameters[key]
            for key in parameters
            if key not in ["self", "kwargs", "func"] #not sure what func parameter is for, seems unecessary to keep
        }
        for key in kwargs: #add all kwargs to dictionary
            parameters[key] = kwargs[key]
        
        self.project = ChildProject(path)
        self.project.read()
        
        try:
            datarepo = Repo(path)
            parameters['dataset_hash'] = datarepo.head.object.hexsha
        except InvalidGitRepositoryError:
            print("Your dataset is not currently a git repository")
            

        if pipeline not in pipelines:
            raise NotImplementedError(f"invalid pipeline '{pipeline}'")

        metrics = pipelines[pipeline](self.project, **kwargs)
        metrics.extract()

        self.metrics = metrics.metrics
        self.metrics.to_csv(self.destination,index=False)
        
        # get the df of metrics used from the Metrics class
        metrics_df = metrics.metrics_list
        metrics_df['callable'] = metrics_df.apply(lambda row: row['callable'].__name__, axis=1) #from the callables used, find their name back
        parameters['metrics_list'] = [ {k:v for k,v in m.items() if pd.notnull(v)} for m in metrics_df.to_dict(orient='records')]
        date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # create a yaml file with all the parameters used
        self.parameters_path = os.path.splitext(self.destination)[0] + "_parameters_{}.yml".format(date)
        print("exported metrics to {}".format(self.destination))
        yaml.dump(
            {
                "package_version": __version__,
                "date": date,
                "parameters": parameters,
            },
            open(self.parameters_path, "w+"),sort_keys=False,
        )
        print("exported sampler parameters to {}".format(self.parameters_path))

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
            choices=["recording_filename", "session_id", "child_id","experiment","segments"],
            default="recording_filename",
        )
        
        parser.add_argument(
            "--segments",
            help="path to a CSV dataframe containing the list of segments to sample from. The CSV should have 3 columns named recording_filename, segment_onset, segment_offset. --by must be set to 'segments', Can not be used along with options [--period,--recordings,--from-tim,--to-time]",
            default=None,
        )
        
        parser.add_argument(
            "--period",
            help="time units to aggregate (optional); equivalent to ``pandas.Grouper`` freq argument. The resulting metrics will be split for each unit across all the resulting periods.",
            default=None,
        )

        parser.add_argument(
            "-f",
            "--from-time",
            help="time range start in HH:MM:SS format (optional)",
            default=None,
        )

        parser.add_argument(
            "-t",
            "--to-time",
            help="time range end in HH:MM:SS format (optional)",
            default=None,
        )
        
        parser.add_argument(
            "--rec-cols",
            help="comma separated columns from recordings.csv to include in the outputted metrics (optional), NA if ambiguous",
            default=None,
        )
        
        parser.add_argument(
            "--child-cols",
            help="comma separated columns from children.csv to include in the outputted metrics (optional), NA if ambiguous",
            default=None,
        )
        
        
        parser.add_argument(
            "--threads", help="amount of threads to run on", default=1, type=int
        )


class MetricsSpecificationPipeline(Pipeline):
    def __init__(self):
        self.metrics = []

    def run(self, parameters_input,func=None):
        #build a dictionary with all parameters used
        parameters = None
        with open(parameters_input, "r") as stream:
            try:
                parameters = yaml.safe_load(stream)
                if 'parameters' in parameters: parameters = parameters['parameters']
            except yaml.YAMLError as exc:
                raise yaml.YAMLError("parsing of the parameters file {} failed. See above exception for more details".format(parameters_input)) from exc
                
        if parameters:
            if "path" not in parameters :
                raise ValueError("the parameter file {} must contain at least the 'path' key specifying the path to the dataset".format(parameters_input))
            if "destination" not in parameters :
                raise ValueError("the parameter file {} must contain the 'destination' key specifying the file to output the metrics to".format(parameters_input))
            if "metrics_list" not in parameters :
                raise ValueError("the parameter file {} must contain the 'metrics_list' key containing the list of the desired metrics".format(parameters_input))
            try:
                metrics_df = pd.DataFrame(parameters["metrics_list"])
            except Exception as e:
                raise ValueError("The 'metrics_list' key in {} must be a list of elements".format(parameters_input)) from e           
        else:
            raise ValueError("could not find any parameters in {}".format(parameters_input))
        
        try:
            datarepo = Repo(parameters["path"])
            parameters['dataset_hash'] = datarepo.head.object.hexsha
        except InvalidGitRepositoryError:
            print("Your dataset is not currently a git repository")
        
        self.project = ChildProject(parameters["path"])
        self.project.read()
        
        self.destination = parameters['destination']
        
        unwanted_keys = {'metrics', 'pipeline'}
        for i in unwanted_keys:
            if i in parameters : del parameters[i]
            
        arguments = {
            key: parameters[key]
            for key in parameters
            if key not in {"metrics_list", "path", "destination","dataset_hash"} 
        }
        try:
            metrics = Metrics(self.project, metrics_df, **arguments)
        except TypeError as e:
            raise ValueError('Unrecognized parameter found {}'.format(e.args[0][46:])) from e
        metrics.extract()

        self.metrics = metrics.metrics
        self.metrics.to_csv(self.destination,index=False)
        
        # get the df of metrics used from the Metrics class
        metrics_df = metrics.metrics_list
        metrics_df['callable'] = metrics_df.apply(lambda row: row['callable'].__name__, axis=1) #from the callables used, find their name back
        parameters['metrics_list'] = [ {k:v for k,v in m.items() if pd.notnull(v)} for m in metrics_df.to_dict(orient='records')]
        date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # create a yaml file with all the parameters used
        self.parameters_path = os.path.splitext(self.destination)[0] + "_parameters_{}.yml".format(date)
        print("exported metrics to {}".format(self.destination))
        yaml.dump(
            {
                "package_version": __version__,
                "date": date,
                "parameters": parameters,
            },
            open(self.parameters_path, "w+"),sort_keys=False,
        )
        print("exported metrics parameters to {}".format(self.parameters_path))

        return self.metrics

    @staticmethod
    def setup_parser(parser):
        parser.add_argument("parameters_input", help="path to the yml file with all parameters")