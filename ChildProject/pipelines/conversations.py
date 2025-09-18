from abc import ABC, abstractmethod
import os
import itertools
import argparse
import datetime
import multiprocessing as mp
import logging
import functools

import numpy as np
import pandas as pd
from typing import Union, List
import yaml
from git import Repo
from git.exc import InvalidGitRepositoryError

import ChildProject
from ChildProject.pipelines.pipeline import Pipeline
from ChildProject.annotations import AnnotationManager

from ChildProject.tables import assert_dataframe, assert_columns_presence, read_csv_with_dtype
import ChildProject.pipelines.conversationFunctions as convfunc
from ..utils import TimeInterval

import time # RM

pipelines = {}

# Create a logger for the module (file)
logger_conversations = logging.getLogger(__name__)
# messages are propagated to the higher level logger (ChildProject), used in cmdline.py
logger_conversations.propagate = True

class Conversations(ABC):
    """
    Main class for generating a conversational extraction from a project object and a list of desired features

    :param project: ChildProject instance of the target dataset.
    :type project: ChildProject.projects.ChildProject
    :param setname: set to extract conversations from (recording_filename, experiment, child_id, session_id, segments), defaults to 'recording_filename', 'segments' is mandatory if passing the segments argument
    :type setname: str
    :param features_list: pandas DataFrame containing the desired features (features functions are in conversationsFunctions.py)
    :type features_list: pd.DataFrame
    :param recordings: recordings to sample from; if None, all recordings will be sampled, defaults to None
    :type recordings: Union[str, List[str], pd.DataFrame], optional
    :param from_time: If specified (in HH:MM:SS format), ignore annotations outside of the given time-range, defaults to None
    :type from_time: str, optional
    :param to_time:  If specified (in HH:MM:SS format), ignore annotations outside of the given time-range, defaults to None
    :type to_time: str, optional
    :param rec_cols: comma separated columns from recordings.csv to include in the outputted extraction (optional), recording_filename,session_id,child_id,duration are always included if possible and dont need to be specified. Any column that is not unique for a given unit (eg date_iso for a child_id being recorded on multiple days) will output a <NA> value
    :type rec_cols: str, optional
    :param child_cols: comma separated columns from children.csv to include in the outputted extraction (optional), None by default
    :type child_cols: str, optional
    :param threads: amount of threads to run on, defaults to 1
    :type threads: int, optional
    """

    def __init__(
            self,
            project: ChildProject.projects.ChildProject,
            setname: str,
            features_list: pd.DataFrame,
            recordings: Union[str, List[str], pd.DataFrame] = None,
            from_time: str = None,
            to_time: str = None,
            rec_cols: str = None, #metadata
            child_cols: str = None, #metadata
            set_cols: str = None,
            threads: int = 1,
    ):

        self.project = project
        self.am = ChildProject.annotations.AnnotationManager(self.project)
        self.threads = int(threads)
        self.conversations = None

        # check that the callable column is either a callable function or a string that can be found as being part of
        # the list of features in ChildProject/pipelines/conversationFunctions.py
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
                 "{} cannot be evaluated as a feature, must be a callable object or a string".format(row["callable"]))

        self.features_df = features_list
        # block checking presence of required columns and evaluates the callable functions
        if isinstance(features_list, pd.DataFrame):
            if ({'callable', 'name'}).issubset(features_list.columns):
                features_list["callable"] = features_list.apply(check_callable, axis=1)
                try:
                    features_list = features_list.set_index('name', verify_integrity=True)
                except ValueError as e:
                    raise ValueError("features_list parameter has duplicates in 'name' column") from e
                features_list['args'] = features_list.drop(['callable'], axis=1).apply(
                    lambda row: row.dropna().to_dict(), axis=1)
                features_list = features_list[['callable', 'args']]
            else:
                raise ValueError("features_list parameter must contain at least the columns [callable,name]")
        else:
            raise ValueError("features_list parameter must be a pandas DataFrame")

        if setname not in self.am.annotations["set"].values:
            raise ValueError(
                f"annotation set '{setname}' was not found in the index; "
                "check spelling and make sure the set was properly imported."
            )
        self.set = setname
        self.features_dict = features_list.to_dict(orient="index")

        # necessary columns to construct the conversations
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
                    logger_conversations.warning(
                        "requested column <{}> does not exist in recordings.csv,\
                         ignoring this column. existing columns are : {}".format(
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
                    logger_conversations.warning(
                        "requested column <{}> does not exist in children.csv, ignoring this column. existing\
                         columns are : {}".format(i, correct_cols))
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

        # get existing columns of the dataset for sets
        correct_cols = set(self.am.sets)
        if set_cols:
            # when user requests recording columns, build the list and verify they exist (warn otherwise)
            set_cols = set(set_cols.split(","))
            for i in set_cols:
                if i not in correct_cols:
                    print(
                        "Warning, requested column <{}> does not exist in the set metadata, ignoring this column. existing columns are : {}".format(
                            i, correct_cols))
            set_cols &= correct_cols
            # add wanted columns to the one we already get
            join_columns.update(set_cols)
        self.set_cols = set_cols

        if recordings is None:
            self.recordings = self.project.recordings['recording_filename'].to_list()
        else:
            self.recordings = Pipeline.recordings_from_list(recordings)

        # turn from_time and to to_time to datetime objects
        if from_time:
            try:
                self.from_time = datetime.datetime.strptime(from_time, "%H:%M:%S")
            except ValueError:
                raise ValueError(
                        f"invalid value for from_time ('{from_time}'); should have HH:MM:SS format instead")
        else:
            self.from_time = None

        if to_time:
            try:
                self.to_time = datetime.datetime.strptime(to_time, "%H:%M:%S")
            except ValueError:
                raise ValueError(f"invalid value for to_time ('{to_time}'); should have HH:MM:SS format instead")
        else:
            self.to_time = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        pipelines[cls.SUBCOMMAND] = cls

    def _process_conversation(self, conversation, rec) -> dict: #process recording line
        """for one conversation block compute the list of required features and store return the results as a dictionary

        :param conversation: index and Series of the unit to process, to be modified with the results
        :type conversation: pd.DataFrame
        :param rec: recording_filename to which belongs that conversation
        :type rec: str
        :return: dict containing all the computed features result for that unit
        :rtype: dict
        """
        segments = conversation

        # results that are included regardless of the required list
        result = {'conversation_onset': segments.iloc[0]['segment_onset'],
                  'conversation_offset': segments['segment_offset'].max(),
                  'voc_count': segments['speaker_type'].count(),
                  'conv_count': conversation.name,
                  'interval_last_conv': conversation.iloc[0]['time_since_last_conv'],
                  'recording_filename': rec,
                  }
        # apply the functions required
        for i in self.features_dict:
            result[i] = self.features_dict[i]["callable"](segments, **self.features_dict[i]['args'])

        return result

    def _process_recording(self, recording, grouper) -> List[dict]:
        """for one recording, get the segments required, group by conversation and launch computation for each block

        :param recording: recording_filename to which belongs that conversation
        :type recording: str
        :return: dict containing all the computed features result for that unit
        :rtype: list[dict]
        """
        segments = self.retrieve_segments(recording)
        segments['voc_duration'] = segments['segment_offset'] - segments['segment_onset']

        # compute the duration between conversation and previous one
        terminals = segments[segments['conv_count'].shift(-1) != segments['conv_count']]
        terminals.index += 1
        steps = (segments[segments['conv_count'].shift(1) != segments['conv_count']]['segment_onset'] -
                 terminals['segment_offset']).dropna()
        steps.index = segments.loc[steps.index, 'conv_count']
        segments['time_since_last_conv'] = segments['conv_count'].map(steps)

        conversations = segments.groupby(grouper, group_keys=True)

        # keep as Series??
        extractions = conversations.apply(
            self._process_conversation, rec=recording).to_list() if len(conversations) else []
        # extractions = [self._process_conversation(block) for block in conversations]

        return extractions

    def extract(self) -> pd.DataFrame:
        """from the initiated self.features_dict, compute each row feature (handles threading)
        Once the Conversation class is initialized, call this function to extract the features and populate
         self.conversations

        :return: DataFrame of computed features
        :rtype: pandas.DataFrame
        """
        grouper = 'conv_count'
        if self.threads == 1:

            results = list(itertools.chain.from_iterable(map(functools.partial(self._process_recording, grouper=grouper), self.recordings)))
        else:
            with mp.Pool(
                    processes=self.threads if self.threads >= 1 else mp.cpu_count()
            ) as pool:
                results = list(itertools.chain.from_iterable(pool.map(functools.partial(self._process_recording, grouper=grouper), self.recordings)))

        self.conversations = pd.DataFrame(results) if len(results) else pd.DataFrame(columns=[grouper])

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
            meta = chis.merge(self.project.recordings[['recording_filename', 'child_id']], how='inner', on='child_id')
            self.conversations = self.conversations.merge(meta, how='left', on='recording_filename')
            if 'child_id' not in self.child_cols:
                self.conversations.drop(columns=['child_id'])
        if self.set_cols:
            for col in self.set_cols:
                if col not in self.conversations.columns:
                    self.conversations[col] = self.am.sets.loc[self.set, col]
                else:
                    logger_conversations.warning(f"Ignoring required column {col} has it already exists in the result")

        if not self.conversations.shape[0]:
            logger_conversations.warning("The extraction did not find any conversation")
        self.conversations = self.conversations.convert_dtypes()
        return self.conversations

    def retrieve_segments(self, recording: str) -> pd.DataFrame:
        """from a list of sets and a row identifying the unit computed, return the relevant annotation segments

        :param recording: recording
        :type recording: str
        :return: relevant annotation segments
        :rtype: pandas.DataFrame
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
            if not segments.shape[0]:
                # no annotations for that unit
                return pd.DataFrame(columns=list(set([c.name for c in AnnotationManager.SEGMENTS_COLUMNS if c.required]
                                         + list(annotations.columns) + ['conv_count'])))
            segments = segments.dropna(subset=['conv_count'])
        else:
            # no annotations for that unit
            return pd.DataFrame(columns=list(set([c.name for c in AnnotationManager.SEGMENTS_COLUMNS if c.required]
                                         + list(annotations.columns) + ['conv_count'])))

        return segments.reset_index(drop=True)


class CustomConversations(Conversations):
    """conversations extraction from a csv file.
    Extracts a number of features listed in a csv file as a dataframe.
    the csv file must contain the columns :
    - 'callable' which is the name of the wanted feature from the list of available features
    - 'name' is the name to give to that feature
    - any other necessary argument for the given feature (eg the is_speaker feature requires the 'speaker' argument: add a column 'speaker' in the csv file and fill its cells for this feature with the wanted value (CHI|FEM|MAL|OCH))

    :param project: ChildProject instance of the target dataset.
    :type project: ChildProject.projects.ChildProject
    :param setname: name of the set to extract conversations from
    :type setname: str
    :param features: name of the csv file listing the features to extract
    :type features: str
    :param recordings: recordings to sample from; if None, all recordings will be sampled, defaults to None
    :type recordings: Union[str, List[str], pd.DataFrame], optional
    :param from_time: If specified (in HH:MM:SS format), ignore annotations outside of the given time-range, defaults to None
    :type from_time: str, optional
    :param to_time:  If specified (in HH:MM:SS format), ignore annotations outside of the given time-range, defaults to None
    :type to_time: str, optional
    :param rec_cols: comma separated columns from recordings.csv to include in the outputted conversations (optional), recording_filename,session_id,child_id,duration are always included if possible and dont need to be specified. Any column that is not unique for a given unit (eg date_iso for a child_id being recorded on multiple days) will output a <NA> value
    :type rec_cols: str, optional
    :param child_cols: comma separated columns from children.csv to include in the outputted conversations (optional), None by default
    :type child_cols: str, optional
    :param threads: amount of threads to run on, defaults to 1
    :type threads: int, optional
    """

    SUBCOMMAND = "custom"

    def __init__(
            self,
            project: ChildProject.projects.ChildProject,
            setname: str,
            features: str,
            recordings: Union[str, List[str], pd.DataFrame] = None,
            from_time: str = None,
            to_time: str = None,
            rec_cols: str = None,
            set_cols: str = None,
            child_cols: str = None,
            threads: int = 1,
    ):
        features_df = pd.read_csv(features)

        super().__init__(project, setname, features_df, recordings=recordings,
                         from_time=from_time, to_time=to_time, rec_cols=rec_cols,
                         child_cols=child_cols, set_cols=set_cols, threads=threads)

    @staticmethod
    def add_parser(subparsers, subcommand):
        parser = subparsers.add_parser(subcommand, help="custom conversation extraction")
        parser.add_argument("features",
                            help="name of the csv file containing the list of features to extract",
                            )


class StandardConversations(Conversations):
    """ACLEW conversations extractor.
    Extracts a number of conversations from the ACLEW pipeline annotations, which includes:

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
    :param rec_cols: comma separated columns from recordings.csv to include in the outputted conversations (optional), recording_filename,session_id,child_id,duration are always included if possible and dont need to be specified. Any column that is not unique for a given unit (eg date_iso for a child_id being recorded on multiple days) will output a <NA> value
    :type rec_cols: str, optional
    :param child_cols: comma separated columns from children.csv to include in the outputted conversations (optional), None by default
    :type child_cols: str, optional
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
            set_cols: str = None,
            threads: int = 1,
    ):

        features = np.array([
             ["who_initiated", "initiator", pd.NA],
             ["who_finished", "finisher", pd.NA],
             ["voc_total_dur", "total_duration_of_vocalisations", pd.NA],
             ["voc_speaker_count", "CHI_voc_count", 'CHI'],
             ["voc_speaker_count", "FEM_voc_count", 'FEM'],
             ["voc_speaker_count", "MAL_voc_count", 'MAL'],
             ["voc_speaker_count", "OCH_voc_count", 'OCH'],
             ["voc_speaker_dur", "CHI_voc_dur", 'CHI'],
             ["voc_speaker_dur", "FEM_voc_dur", 'FEM'],
             ["voc_speaker_dur", "MAL_voc_dur", 'MAL'],
             ["voc_speaker_dur", "OCH_voc_dur", 'OCH'],
             ])

        features = pd.DataFrame(features, columns=["callable", "name", "speaker"])

        super().__init__(project, setname, features, recordings=recordings,
                         from_time=from_time, to_time=to_time,
                         rec_cols=rec_cols, child_cols=child_cols,
                         set_cols=set_cols, threads=threads)

    @staticmethod
    def add_parser(subparsers, subcommand):
        parser = subparsers.add_parser(subcommand, help="standard conversation extraction")


class ConversationsPipeline(Pipeline):
    def __init__(self):
        self.destination = None
        self.project = None
        self.conversations = None
        self.parameters_path = None

    def run(self, path, destination, pipeline, func=None, **kwargs) -> pd.DataFrame:
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
            logger_conversations.warning("Your dataset is not currently a git repository")

        if pipeline not in pipelines:
            raise NotImplementedError(f"invalid pipeline '{pipeline}'")

        conversations = pipelines[pipeline](self.project, **kwargs)
        conversations.extract()

        self.conversations = conversations.conversations
        self.conversations.to_csv(self.destination, index=False)

        # get the df of features used from the Conversations class
        features_df = conversations.features_df
        features_df['callable'] = features_df.apply(lambda row: row['callable'].__name__,
                                                  axis=1)  # from the callables used, find their name back
        parameters['features_list'] = [{k: v for k, v in m.items() if pd.notnull(v)} for m in
                                       features_df.to_dict(orient='records')]
        date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # create a yaml file with all the parameters used
        self.parameters_path = os.path.splitext(self.destination)[0] + "_parameters_{}.yml".format(date)
        logger_conversations.info("exported conversations to {}".format(self.destination))
        yaml.dump(
            {
                "package_version": ChildProject.__version__,
                "date": date,
                "parameters": parameters,
            },
            open(self.parameters_path, "w+"), sort_keys=False,
        )
        logger_conversations.info("exported sampler parameters to {}".format(self.parameters_path))

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
            help=("path to a CSV dataframe containing the list of recordings to sample from (by default, all recordings"
                  " will be sampled). The CSV should have one column named recording_filename."),
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
            help=("comma separated columns from recordings.csv to include in the outputted conversations (optional),"
                  " NA if ambiguous"),
            default=None,
        )

        parser.add_argument(
            "--child-cols",
            help=("comma separated columns from children.csv to include in the outputted conversations (optional),"
                  " NA if ambiguous"),
            default=None,
        )

        parser.add_argument(
            "--set-cols",
            help="comma separated columns from the set metadata to include in the outputted conversations (optional)",
            default=None,
        )

        parser.add_argument(
            "--threads", help="amount of threads to run on", default=1, type=int
        )


class ConversationsSpecificationPipeline(Pipeline):
    def __init__(self):
        self.destination = None
        self.project = None
        self.conversations = None
        self.parameters_path = None

    def run(self, parameters_input, func=None) -> pd.DataFrame:
        # build a dictionary with all parameters used
        parameters = None
        with open(parameters_input, "r") as stream:
            try:
                parameters = yaml.safe_load(stream)
                if 'parameters' in parameters:
                    parameters = parameters['parameters']
            except yaml.YAMLError as exc:
                raise yaml.YAMLError(
                    "parsing of the parameters file {} failed. See above exception for more details".format(
                        parameters_input)) from exc

        if parameters:
            if "path" not in parameters:
                raise ValueError(
                    ("the parameter file {} must contain at least the 'path' key specifying the path to the "
                     "dataset".format(parameters_input)))
            if "destination" not in parameters:
                raise ValueError(
                    ("the parameter file {} must contain the 'destination' key specifying the file to output "
                     "the conversations to".format(parameters_input)))
            if "features_list" not in parameters:
                raise ValueError(
                    ("the parameter file {} must contain the 'features_list' key containing the list of the desired "
                     "features".format(parameters_input)))
            try:
                features_df = pd.DataFrame(parameters["features_list"])
            except Exception as e:
                raise ValueError(
                    "The 'features_list' key in {} must be a list of elements".format(parameters_input)) from e
        else:
            raise ValueError("could not find any parameters in {}".format(parameters_input))

        try:
            datarepo = Repo(parameters["path"])
            parameters['dataset_hash'] = datarepo.head.object.hexsha
        except InvalidGitRepositoryError:
            logger_conversations.warning("Your dataset is not currently a git repository")

        self.project = ChildProject.projects.ChildProject(parameters["path"])
        self.project.read()

        self.destination = parameters['destination']

        unwanted_keys = {'features', 'pipeline'}
        for i in unwanted_keys:
            if i in parameters:
                del parameters[i]

        arguments = {
            key: parameters[key]
            for key in parameters
            if key not in {"features_list", "path", "destination", "dataset_hash"}
        }
        try:
            conversations = Conversations(self.project, features_list=features_df, **arguments)
        except TypeError as e:
            raise ValueError('Unrecognized parameter found {}'.format(e.args[0][46:])) from e
        conversations.extract()

        self.conversations = conversations.conversations
        self.conversations.to_csv(self.destination, index=False)

        # get the df of features used from the Conversations class
        features_df = conversations.features_df
        print(features_df)
        features_df['callable'] = features_df.apply(lambda row: row['callable'].__name__,
                                                    axis=1)  # from the callables used, find their name back
        parameters['features_list'] = [{k: v for k, v in m.items() if pd.notnull(v)} for m in
                                       features_df.to_dict(orient='records')]
        date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # create a yaml file with all the parameters used
        self.parameters_path = os.path.splitext(self.destination)[0] + "_parameters_{}.yml".format(date)
        logger_conversations.info("exported conversations to {}".format(self.destination))
        yaml.dump(
            {
                "package_version": ChildProject.__version__,
                "date": date,
                "parameters": parameters,
            },
            open(self.parameters_path, "w+"), sort_keys=False,
        )
        logger_conversations.info("exported conversations parameters to {}".format(self.parameters_path))

        return self.conversations

    @staticmethod
    def setup_parser(parser):
        parser.add_argument("parameters_input", help="path to the yml file with all parameters")
