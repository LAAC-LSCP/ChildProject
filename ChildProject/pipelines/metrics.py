from abc import ABC, abstractmethod
import argparse
import datetime
import multiprocessing as mp
import numpy as np
import pandas as pd
from typing import Union, List

import ChildProject
from ChildProject.pipelines.pipeline import Pipeline

pipelines = {}

class Metrics(ABC):
    def __init__(self,
        project: ChildProject.projects.ChildProject,
        by: str = 'recording_filename',
        recordings: Union[str, List[str], pd.DataFrame] = None):

        self.project = project
        self.am = ChildProject.annotations.AnnotationManager(self.project)

        self.am.annotations = self.am.annotations.merge(
            self.project.recordings[['recording_filename', 'session_id', 'session_offset', 'child_id', 'duration']],
            left_on = 'recording_filename',
            right_on = 'recording_filename'
        )

        self.by = by
        self.segments = pd.DataFrame()

        if recordings is None:
            self.recordings = None
        elif isinstance(recordings, pd.DataFrame):
            if 'recording_filename' not in recordings.columns:
                raise ValueError("recordings dataframe is missing a 'recording_filename' column")
            self.recordings = recordings['recording_filename'].tolist()
        elif isinstance(recordings, pd.Series):
            self.recordings = recordings.tolist()
        elif isinstance(recordings, list):
            self.recordings = recordings
        else:
            if not os.path.exists(recordings):
                raise ValueError(
                    "'recordings' is neither a pandas dataframe,"
                    "nor a list or a path to an existing dataframe."
                )
            
            df = pd.read_csv(recordings)
            if 'recording_filename' not in df.columns:
                raise ValueError(
                    f"'{recordings}' is missing a 'recording_filename' column"
                )
            self.recordings = df['recording_filename'].tolist()

        if self.recordings is not None:
            self.recordings = list(set(self.recordings))

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        pipelines[cls.SUBCOMMAND] = cls

    @abstractmethod
    def extract(self):
        pass

    def get_recordings(self):
        recordings = self.project.recordings.copy()

        if self.recordings is not None:
            recordings = recordings[recordings['recording_filename'].isin(self.recordings)]
        
        return recordings

    def retrieve_segments(self, set: str, unit: str):
        annotations = self.am.annotations[self.am.annotations[self.by] == unit]
        annotations = annotations[annotations['set'] == set]
        
        try:
            segments = self.am.get_segments(annotations)
        except Exception as e:
            print(str(e))
            return pd.DataFrame()

        # prevent overflows
        segments['segment_onset'] /= 1000
        segments['segment_offset'] /= 1000

        segments['duration'] = (segments['segment_offset']-segments['segment_onset']).astype(float).fillna(0)
        
        return segments

class LenaMetrics(Metrics):
    SUBCOMMAND = 'lena'

    def __init__(self,
        project: ChildProject.projects.ChildProject,
        set: str,
        recordings: Union[str, List[str], pd.DataFrame],
        by: str = 'recording_filename',
        threads: int = 1):

        super().__init__(project, by, recordings)

        self.set = set
        self.threads = int(threads)

    def _process_unit(self, unit: str):
        import ast

        its = self.retrieve_segments(self.set, unit)
        if len(its) == 0:
            return pd.DataFrame()

        unit_duration = self.project.recordings[self.project.recordings[self.by] == unit]['duration'].sum()/1000

        metrics = {}
        speaker_types = ['FEM', 'MAL', 'CHI', 'OCH']
        adults = ['FEM', 'MAL']
        
        its = its[its['speaker_type'].isin(speaker_types)]

        if len(its) == 0:
            return pd.DataFrame()

        its_agg = its.groupby('speaker_type').agg(
            voc_ph = ('segment_onset', lambda x: 3600*len(x)/unit_duration),
            voc_dur_ph = ('duration', lambda x: 3600*np.sum(x)/unit_duration),
            avg_voc_dur = ('duration', np.mean),
            wc_ph = ('words', lambda x: 3600*np.sum(x)/unit_duration)
        )

        for speaker in speaker_types:
            if speaker not in its_agg.index:
                continue

            metrics['voc_{}_ph'.format(speaker.lower())] = its_agg.loc[speaker, 'voc_ph']
            metrics['voc_dur_{}_ph'.format(speaker.lower())] = its_agg.loc[speaker, 'voc_dur_ph']
            metrics['avg_voc_dur_{}'.format(speaker.lower())] = its_agg.loc[speaker, 'avg_voc_dur']

            if speaker in adults:
                metrics['wc_{}_ph'.format(speaker.lower())] = its_agg.loc[speaker, 'wc_ph']

        chi = its[its['speaker_type'] == 'CHI']
        cries = chi['cries'].apply(lambda x: len(ast.literal_eval(x))).sum()
        vfxs = chi['vfxs'].apply(lambda x: len(ast.literal_eval(x))).sum()
        utterances = chi['utterances_count'].sum()

        metrics['lp_n'] = utterances/(utterances + cries + vfxs)
        metrics['lp_dur'] = chi['utterances_length'].sum()/(chi['child_cry_vfx_len'].sum()+chi['utterances_length'].sum())

        metrics['wc_adu_ph'] = its['words'].sum()*3600/unit_duration

        metrics[self.by] = unit
        metrics['child_id'] = its['child_id'].iloc[0]
        metrics['duration'] = unit_duration

        return pd.DataFrame([metrics])

    
    def extract(self):
        recordings = self.get_recordings()

        if self.threads == 1:
            self.metrics = pd.concat([self._process_unit(unit) for unit in recordings[self.by].unique()])
        else:
            with mp.Pool(processes = self.threads if self.threads >= 1 else mp.cpu_count()) as pool:
                self.metrics = pd.concat(pool.map(self._process_unit, recordings[self.by].unique()))

        self.metrics.set_index(self.by, inplace = True)
        return self.metrics
        

    @staticmethod
    def add_parser(subparsers, subcommand):
        parser = subparsers.add_parser(subcommand, help = 'LENA metrics')
        parser.add_argument('set', help = 'name of the LENA its annotations set')
        parser.add_argument('--threads', help = 'amount of threads to run on', default = 1, type = int)

class MetricsPipeline(Pipeline):
    def __init__(self):
        self.metrics = []

    def run(self, path, destination, pipeline, func = None, **kwargs):
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
        parser.add_argument('path', help = 'path to the dataset')
        parser.add_argument('destination', help = 'segments destination')

        subparsers = parser.add_subparsers(help = 'pipeline', dest = 'pipeline')
        for pipeline in pipelines:
            pipelines[pipeline].add_parser(subparsers, pipeline)

        parser.add_argument('--recordings',
            help = "path to a CSV dataframe containing the list of recordings to sample from (by default, all recordings will be sampled). The CSV should have one column named recording_filename.",
            default = None
        )

        parser.add_argument(
            '--by',
            help = 'units to sample from (default behavior is to sample by recording)',
            choices = ['recording_filename', 'session_id', 'child_id'],
            default = 'recording_filename'
        )
