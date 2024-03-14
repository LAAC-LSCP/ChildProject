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

import ChildProject
from ChildProject.pipelines.pipeline import Pipeline

from ChildProject.tables import assert_dataframe, assert_columns_presence, read_csv_with_dtype
import ChildProject.pipelines.conversationFunctions as convfunc
from ..utils import TimeInterval

pipelines = {}


class Conversations(ABC):
    """
    Main class for generating conversational metrics from a project object and a list of desired metrics

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
    :param threads: amount of threads to run on, defaults to 1
    :type threads: int, optional
    """

    def __init__(
            self,
            project: ChildProject.projects.ChildProject,
            setname: str,
            metrics_list: pd.DataFrame,
            recordings: Union[str, List[str], pd.DataFrame] = None,
            from_time: str = None,
            to_time: str = None,
            rec_cols: str = None, #metadata
            child_cols: str = None, #metadata
            threads: int = 1,
    ):

        self.project = project
        self.am = ChildProject.annotations.AnnotationManager(self.project)
        self.threads = int(threads)
        self.conversations = None

        # check that the callable column is either a callable function or a string that can be found as being part of
        # the list of metrics in ChildProject/pipelines/conversationFunctions.py
        def check_callable(row):
            if callable(row["callable"]): return row["callable"]
            if isinstance(row["callable"], str):
                try:
                    f = getattr(convfunc, row["callable"])
                except Exception:
                    raise ValueError(
                        "{} function is not defined and was not found in ChildProject/pipelines/conversationFunctions.py".format(
                            row["callable"]))
                return f
            else:
                raise ValueError(
                    "{} cannot be evaluated as a metric, must be a callable object or a string".format(row["callable"]))

        # block checking presence of required columns and evaluates the callable functions
        if isinstance(metrics_list, pd.DataFrame):
            if ({'callable', 'name'}).issubset(metrics_list.columns):
                metrics_list["callable"] = metrics_list.apply(check_callable, axis=1)
            else:
                raise ValueError("metrics_list parameter must contain at least the columns [callable,name]")
        else:
            raise ValueError("metrics_list parameter must be a pandas DataFrame")

        if setname not in self.am.annotations["set"].values:
            raise ValueError(
                f"annotation set '{setname}' was not found in the index; "
                "check spelling and make sure the set was properly imported."
            )
        self.set = setname
        self.metrics_list = metrics_list

        # necessary columns to construct the metrics
        join_columns = {
            "recording_filename",
            "child_id",
            "duration",
            "session_id",
            "session_offset",
        }
        # get existing columns of the dataset for recordings
        correct_cols = set(self.project.recordings.columns)

        if rec_cols:
            # when user requests recording columns, build the list and verify they exist (warn otherwise)
            rec_cols = set(rec_cols.split(","))
            for i in rec_cols:
                if i not in correct_cols:
                    print(
                        "Warning, requested column <{}> does not exist in recordings.csv, ignoring this column. existing columns are : {}".format(
                            i, correct_cols))
            rec_cols &= correct_cols
            # add wanted columns to the one we already get
            join_columns.update(rec_cols)
        self.rec_cols = rec_cols

        join_columns &= correct_cols

        # join dataset annotation with their info in recordings.csv
        self.am.annotations = self.am.annotations.merge(
            self.project.recordings[list(join_columns)],
            left_on="recording_filename",
            right_on="recording_filename",
        )

        # get existing columns of the dataset for children
        correct_cols = set(self.project.children.columns)
        if child_cols:
            # when user requests children columns, build the list and verify they exist (warn otherwise)
            child_cols = set(child_cols.split(","))
            child_cols.add("child_id")
            for i in child_cols:
                if i not in correct_cols:
                    print(
                        "Warning, requested column <{}> does not exist in children.csv, ignoring this column. existing columns are : {}".format(
                            i, correct_cols))
            child_cols &= correct_cols
            self.child_cols = child_cols

            # join dataset annotation with their info in children.csv
            self.am.annotations = self.am.annotations.merge(
                self.project.children[list(child_cols)],
                left_on="child_id",
                right_on="child_id",
            )
        else:
            self.child_cols = None

        if recordings is None:
            self.recordings = self.project.recordings['recording_filename'].to_list()
        else:
            self.recordings = Pipeline.recordings_from_list(recordings)

        # turn from_time and to to_time to datetime objects
        if from_time:
            try:
                self.from_time = datetime.datetime.strptime(from_time, "%H:%M:%S")
            except:
                raise ValueError(
                        f"invalid value for from_time ('{from_time}'); should have HH:MM:SS format instead")
        else:
            self.from_time = None

        if to_time:
            try:
                self.to_time = datetime.datetime.strptime(to_time, "%H:%M:%S")
            except:
                raise ValueError(f"invalid value for to_time ('{to_time}'); should have HH:MM:SS format instead")
        else:
            self.to_time = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        pipelines[cls.SUBCOMMAND] = cls

    def _process_conversation(self, conversation): #process recording line
        #keep lines for which conv_count is nopt Na and group by conv
        """for one unit (i.e. 1 recording) compute the list of required metrics and store the results in the current row of self.metrics

        :param row: index and Series of the unit to process, to be modified with the results
        :type row: (int , pandas.Series)
        :return: Series containing all the computed metrics result for that unit
        :rtype: pandas.Series
        """
        meta, annotations = conversation
        result = {'recording_filename': meta[0], 'conv_count': meta[1]}
        for i, line in self.metrics_list.iterrows():

            annotations['voc_duration'] = annotations['segment_offset'] - annotations['segment_onset']

            result[line['name']] = line["callable"](annotations, **line.drop(['callable', 'name']).dropna().to_dict())

        return result

    def extract(self):
        """from the initiated self.metrics, compute each row metrics (handles threading)
        Once the Metrics class is initialized, call this function to extract the metrics and populate self.metrics

        :return: DataFrame of computed metrics
        :rtype: pandas.DataFrame
        """
        if self.threads == 1:
            full_annotations = pd.concat([self.retrieve_segments(rec) for rec in self.recordings])
        else:
            with mp.Pool(
                    processes=self.threads if self.threads >= 1 else mp.cpu_count()
            ) as pool:
                full_annotations = pd.concat(pool.map(self.retrieve_segments, self.recordings))

        conversations = full_annotations.groupby(['recording_filename', 'conv_count'])

        if self.threads == 1:
            self.conversations = pd.DataFrame(
                [self._process_conversation(block) for block in conversations]
            )
        else:
            with mp.Pool(
                    processes=self.threads if self.threads >= 1 else mp.cpu_count()
            ) as pool:
                self.conversations = pd.DataFrame(
                    pool.map(self._process_conversation, [block for block in conversations])
                )

        # now add the rec_cols and child_cols in the result
        if self.rec_cols:
            if self.child_cols:
                recs = self.project.recordings.drop(columns=(
                    [col for col in self.project.recordings.columns if (col not in self.rec_cols
                                                                        and col != 'recording_filename'
                                                                        and col != 'child_id')]
                ))
                chis = self.project.children.drop(columns=(
                    [col for col in self.project.children.columns if (col not in self.child_cols
                                                                      and col != 'child_id')]
                ))
                meta = recs.merge(chis, how='inner', on='child_id')
                self.conversations = self.conversations.merge(meta, how='left', on='recording_filename')
                if 'child_id' not in self.child_cols and 'child_id' not in self.rec_cols:
                    self.conversations.drop(columns=['child_id'])
            else:
                recs = self.project.recordings.drop(columns=(
                    [col for col in self.project.recordings.columns if (col not in self.rec_cols
                                                                        and col != 'recording_filename'
                                                                        and col != 'child_id')]
                ))
                self.conversations = self.conversations.merge(recs, how='left', on='recording_filename')
        elif self.child_cols:
            chis = self.project.children.drop(columns=(
                [col for col in self.project.children.columns if (col not in self.child_cols
                                                                  and col != 'child_id')]
            ))
            meta = chis.merge(self.project.recordings[['recording_filename','child_id']], how='inner', on='child_id')
            self.conversations = self.conversations.merge(meta, how='left', on='recording_filename')
            if 'child_id' not in self.child_cols:
                self.conversations.drop(columns=['child_id'])

        return self.conversations

    def retrieve_segments(self, recording: str):
        """from a list of sets and a row identifying the unit computed, return the relevant annotation segments

        :param sets: List of annotation sets to keep
        :type sets: List[str]
        :param row: Series storing the unit to compute information
        :type row: pandas.Series
        :return: relevant annotation DataFrame and index DataFrame
        :rtype: (pandas.DataFrame , pandas.DataFrame)
        """
        annotations = self.am.annotations[recording == self.am.annotations['recording_filename']]
        annotations = annotations[annotations["set"] == self.set]
        # restrict to time ranges
        if self.from_time and self.to_time:
            matches = self.am.get_within_time_range(
                annotations, TimeInterval(self.from_time, self.to_time))
        else:
            matches = annotations

        if matches.shape[0]:
            segments = self.am.get_segments(matches)
            segments = segments.dropna(subset='conv_count')
        else:
            # no annotations for that unit
            return pd.DataFrame(), pd.DataFrame()

        segments['recording_filename'] = recording

        #TODO check that required columns exist

        return segments


class CustomConversations(Conversations):
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
    :param threads: amount of threads to run on, defaults to 1
    :type threads: int, optional
    """

    SUBCOMMAND = "custom"

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
            threads: int = 1,
    ):
        metrics_df = pd.read_csv(metrics)

        super().__init__(project, metrics_df, by=by, recordings=recordings,
                         from_time=from_time, to_time=to_time, rec_cols=rec_cols,
                         child_cols=child_cols, threads=threads)

    @staticmethod
    def add_parser(subparsers, subcommand):
        pass


class StandardConversations(Conversations):
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
    :param threads: amount of threads to run on, defaults to 1
    :type threads: int, optional
    """

    SUBCOMMAND = "standard"

    def __init__(
            self,
            project: ChildProject.projects.ChildProject,
            setname: str = "vtc/conversations",
            recordings: Union[str, List[str], pd.DataFrame] = None,
            from_time: str = None,
            to_time: str = None,
            rec_cols: str = None,
            child_cols: str = None,
            threads: int = 1,
    ):

        METRICS = np.array(
            [["conversation_onset", "conversation_onset", pd.NA],
             ["conversation_offset", "conversation_offset", pd.NA],
             ["conversation_duration", "conversation_duration", pd.NA],
             ["vocalisations_count", "vocalisations_count", pd.NA],
             ["who_initiated", "initiator", pd.NA],
             ["who_finished", "finisher", pd.NA],
             ["who_participates", "participators", pd.NA],
             ["total_duration_of_vocalisations", "total_duration_of_vocalisations", pd.NA],
             ["conversation_duration", "conversation_duration", pd.NA],
             ["is_speaker", "CHI_present", 'CHI'],
             ["is_speaker", "FEM_present", 'FEM'],
             ["is_speaker", "MAL_present", 'MAL'],
             ["is_speaker", "OCH_present", 'OCH'],
             ["voc_counter", "CHI_voc_counter", 'CHI'],
             ["voc_counter", "FEM_voc_counter", 'FEM'],
             ["voc_counter", "MAL_voc_counter", 'MAL'],
             ["voc_counter", "OCH_voc_counter", 'OCH'],
             ["voc_total", "CHI_voc_total", 'CHI'],
             ["voc_total", "FEM_voc_total", 'FEM'],
             ["voc_total", "MAL_voc_total", 'MAL'],
             ["voc_total", "OCH_voc_total", 'OCH'],
             ["voc_contribution", "CHI_voc_contribution", 'CHI'],
             ["voc_contribution", "FEM_voc_contribution", 'FEM'],
             ["voc_contribution", "MAL_voc_contribution", 'MAL'],
             ["voc_contribution", "OCH_voc_contribution", 'OCH'],
             ["assign_conv_type", "conversation_type", pd.NA],
             ])

        METRICS = pd.DataFrame(METRICS, columns=["callable", "name", "speaker"])

        super().__init__(project, setname, METRICS, recordings=recordings,
                         from_time=from_time, to_time=to_time,
                         rec_cols=rec_cols, child_cols=child_cols,
                         threads=threads)

    @staticmethod
    def add_parser(subparsers, subcommand):
        parser = subparsers.add_parser(subcommand, help="standard conversation extraction")


class ConversationsPipeline(Pipeline):
    def __init__(self):
        self.metrics = []

    def run(self, path, destination, pipeline, func=None, **kwargs):
        self.destination = destination
        # build a dictionary with all parameters used
        parameters = locals()
        parameters = {
            key: parameters[key]
            for key in parameters
            if key not in ["self", "kwargs", "func"]  # not sure what func parameter is for, seems unecessary to keep
        }
        for key in kwargs:  # add all kwargs to dictionary
            parameters[key] = kwargs[key]

        self.project = ChildProject.projects.ChildProject(path)
        self.project.read()

        try:
            datarepo = Repo(path)
            parameters['dataset_hash'] = datarepo.head.object.hexsha
        except InvalidGitRepositoryError:
            print("Your dataset is not currently a git repository")

        if pipeline not in pipelines:
            raise NotImplementedError(f"invalid pipeline '{pipeline}'")

        conversations = pipelines[pipeline](self.project, **kwargs)
        conversations.extract()

        self.conversations = conversations.conversations
        self.conversations.to_csv(self.destination, index=False)

        # get the df of metrics used from the Metrics class
        metrics_df = conversations.metrics_list
        metrics_df['callable'] = metrics_df.apply(lambda row: row['callable'].__name__,
                                                  axis=1)  # from the callables used, find their name back
        parameters['metrics_list'] = [{k: v for k, v in m.items() if pd.notnull(v)} for m in
                                      metrics_df.to_dict(orient='records')]
        date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # create a yaml file with all the parameters used
        self.parameters_path = os.path.splitext(self.destination)[0] + "_parameters_{}.yml".format(date)
        print("exported metrics to {}".format(self.destination))
        yaml.dump(
            {
                "package_version": ChildProject.__version__,
                "date": date,
                "parameters": parameters,
            },
            open(self.parameters_path, "w+"), sort_keys=False,
        )
        print("exported sampler parameters to {}".format(self.parameters_path))

        return self.conversations

    @staticmethod
    def setup_parser(parser):
        parser.add_argument("path", help="path to the dataset")
        parser.add_argument("destination", help="segments destination")

        subparsers = parser.add_subparsers(help="pipeline", dest="pipeline")
        for pipeline in pipelines:
            pipelines[pipeline].add_parser(subparsers, pipeline)

        parser.add_argument(
            "--set",
            help="Set to use to get the conversation annotations",
            required=True,
            dest='setname'
        )

        parser.add_argument(
            "--recordings",
            help="path to a CSV dataframe containing the list of recordings to sample from (by default, all recordings will be sampled). The CSV should have one column named recording_filename.",
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


class ConversationsSpecificationPipeline(Pipeline):
    def __init__(self):
        self.metrics = []

    def run(self, parameters_input, func=None):
        # build a dictionary with all parameters used
        parameters = None
        with open(parameters_input, "r") as stream:
            try:
                parameters = yaml.safe_load(stream)
                if 'parameters' in parameters: parameters = parameters['parameters']
            except yaml.YAMLError as exc:
                raise yaml.YAMLError(
                    "parsing of the parameters file {} failed. See above exception for more details".format(
                        parameters_input)) from exc

        if parameters:
            if "path" not in parameters:
                raise ValueError(
                    "the parameter file {} must contain at least the 'path' key specifying the path to the dataset".format(
                        parameters_input))
            if "destination" not in parameters:
                raise ValueError(
                    "the parameter file {} must contain the 'destination' key specifying the file to output the metrics to".format(
                        parameters_input))
            if "metrics_list" not in parameters:
                raise ValueError(
                    "the parameter file {} must contain the 'metrics_list' key containing the list of the desired metrics".format(
                        parameters_input))
            try:
                metrics_df = pd.DataFrame(parameters["metrics_list"])
            except Exception as e:
                raise ValueError(
                    "The 'metrics_list' key in {} must be a list of elements".format(parameters_input)) from e
        else:
            raise ValueError("could not find any parameters in {}".format(parameters_input))

        try:
            datarepo = Repo(parameters["path"])
            parameters['dataset_hash'] = datarepo.head.object.hexsha
        except InvalidGitRepositoryError:
            print("Your dataset is not currently a git repository")

        self.project = ChildProject.projects.ChildProject(parameters["path"])
        self.project.read()

        self.destination = parameters['destination']

        unwanted_keys = {'metrics', 'pipeline'}
        for i in unwanted_keys:
            if i in parameters: del parameters[i]

        arguments = {
            key: parameters[key]
            for key in parameters
            if key not in {"metrics_list", "path", "destination", "dataset_hash"}
        }
        try:
            conversations = Conversations(self.project, metrics_df, **arguments)
        except TypeError as e:
            raise ValueError('Unrecognized parameter found {}'.format(e.args[0][46:])) from e
        conversations.extract()

        self.conversations = conversations.conversations
        self.conversations.to_csv(self.destination, index=False)

        # get the df of metrics used from the Metrics class
        metrics_df = conversations.metrics_list
        metrics_df['callable'] = metrics_df.apply(lambda row: row['callable'].__name__,
                                                  axis=1)  # from the callables used, find their name back
        parameters['metrics_list'] = [{k: v for k, v in m.items() if pd.notnull(v)} for m in
                                      metrics_df.to_dict(orient='records')]
        date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # create a yaml file with all the parameters used
        self.parameters_path = os.path.splitext(self.destination)[0] + "_parameters_{}.yml".format(date)
        print("exported metrics to {}".format(self.destination))
        yaml.dump(
            {
                "package_version": ChildProject.__version__,
                "date": date,
                "parameters": parameters,
            },
            open(self.parameters_path, "w+"), sort_keys=False,
        )
        print("exported metrics parameters to {}".format(self.parameters_path))

        return self.conversations

    @staticmethod
    def setup_parser(parser):
        parser.add_argument("parameters_input", help="path to the yml file with all parameters")