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

        recording_columns = {
            'recording_filename', 'child_id', 'duration', 'session_id', 'session_offset'
        }
        recording_columns &= set(self.project.recordings.columns)

        self.am.annotations = self.am.annotations.merge(
            self.project.recordings[recording_columns],
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

    def retrieve_segments(self, sets: List[str], unit: str):
        annotations = self.am.annotations[self.am.annotations[self.by] == unit]
        annotations = annotations[annotations['set'].isin(sets)]
        
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
    """LENA metrics extractor. 
    Extracts a number of metrics from the LENA .its annotations.

    :param project: ChildProject instance of the target dataset.
    :type project: ChildProject.projects.ChildProject
    :param set: name of the set associated to the .its annotations
    :type set: str
    :param recordings: recordings to sample from; if None, all recordings will be sampled, defaults to None
    :type recordings: Union[str, List[str], pd.DataFrame], optional
    :param by: units to sample from, defaults to 'recording_filename'
    :type by: str, optional
    :param threads: amount of threads to run on, defaults to 1
    :type threads: int, optional
    """

    SUBCOMMAND = 'lena'

    def __init__(self,
        project: ChildProject.projects.ChildProject,
        set: str,
        recordings: Union[str, List[str], pd.DataFrame] = None,
        by: str = 'recording_filename',
        threads: int = 1):

        super().__init__(project, by, recordings)

        self.set = set
        self.threads = int(threads)

    def _process_unit(self, unit: str):
        import ast

        its = self.retrieve_segments([self.set], unit)
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

        return metrics

    
    def extract(self):
        recordings = self.get_recordings()

        if self.threads == 1:
            self.metrics = pd.DataFrame([self._process_unit(unit) for unit in recordings[self.by].unique()])
        else:
            with mp.Pool(processes = self.threads if self.threads >= 1 else mp.cpu_count()) as pool:
                self.metrics = pd.DataFrame(pool.map(self._process_unit, recordings[self.by].unique()))

        self.metrics.set_index(self.by, inplace = True)
        return self.metrics
        

    @staticmethod
    def add_parser(subparsers, subcommand):
        parser = subparsers.add_parser(subcommand, help = 'LENA metrics')
        parser.add_argument('set', help = 'name of the LENA its annotations set')
        parser.add_argument('--threads', help = 'amount of threads to run on', default = 1, type = int)

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
    :param by: units to sample from, defaults to 'recording_filename'
    :type by: str, optional
    :param threads: amount of threads to run on, defaults to 1
    :type threads: int, optional
    """

    SUBCOMMAND = 'aclew'

    def __init__(self,
        project: ChildProject.projects.ChildProject,
        vtc: str = 'vtc',
        alice: str = 'alice',
        vcm: str = 'vcm',
        recordings: Union[str, List[str], pd.DataFrame] = None,
        by: str = 'recording_filename',
        threads: int = 1):

        super().__init__(project, by, recordings)

        self.vtc = vtc
        self.alice = alice
        self.vcm = vcm
        self.threads = int(threads)

    def _process_unit(self, unit: str):
        segments = self.retrieve_segments([self.vtc, self.alice, self.vcm], unit)

        if len(segments) == 0:
            return pd.DataFrame()

        unit_duration = self.project.recordings[self.project.recordings[self.by] == unit]['duration'].sum()/1000

        metrics = {}
        speaker_types = ['FEM', 'MAL', 'CHI', 'OCH']
        adults = ['FEM', 'MAL']

        segments = segments[segments['speaker_type'].isin(speaker_types)]

        vtc = segments[segments['set'] == self.vtc]
        alice = segments[segments['set'] == self.alice]
        vcm = segments[segments['set'] == self.vcm]

        vtc_agg = vtc.groupby('speaker_type').agg(
            voc_ph = ('segment_onset', lambda x: 3600*len(x)/unit_duration),
            voc_dur_ph = ('duration', lambda x: 3600*np.sum(x)/unit_duration),
            avg_voc_dur = ('duration', np.mean)
        )

        for speaker in speaker_types:
            if speaker not in vtc_agg.index:
                continue

            metrics['voc_{}_ph'.format(speaker.lower())] = vtc_agg.loc[speaker, 'voc_ph']
            metrics['voc_dur_{}_ph'.format(speaker.lower())] = vtc_agg.loc[speaker, 'voc_dur_ph']
            metrics['avg_voc_dur_{}'.format(speaker.lower())] = vtc_agg.loc[speaker, 'avg_voc_dur']

        if len(alice):
            alice_agg = alice.groupby('speaker_type').agg(
                wc_ph = ('words', lambda x: 3600*np.sum(x)/unit_duration),
                sc_ph = ('syllables', lambda x: 3600*np.sum(x)/unit_duration),
                pc_ph = ('phonemes', lambda x: 3600*np.sum(x)/unit_duration)
            )

            for speaker in adults:
                if speaker not in alice_agg.index:
                    continue
            
                metrics['wc_{}_ph'.format(speaker.lower())] = alice_agg.loc[speaker, 'wc_ph']
                metrics['sc_{}_ph'.format(speaker.lower())] = alice_agg.loc[speaker, 'sc_ph']
                metrics['pc_{}_ph'.format(speaker.lower())] = alice_agg.loc[speaker, 'pc_ph']

            metrics['wc_adu_ph'] = alice['words'].sum()*3600/unit_duration
            metrics['sc_adu_ph'] = alice['syllables'].sum()*3600/unit_duration
            metrics['pc_adu_ph'] = alice['phonemes'].sum()*3600/unit_duration

        if len(vcm):
            vcm_agg = vcm[vcm['speaker_type'] == 'CHI'].groupby('vcm_type').agg(
                voc_chi_ph = ('segment_onset', lambda x: 3600*len(x)/unit_duration),
                voc_dur_chi_ph = ('duration', lambda x: 3600*np.sum(x)/unit_duration),
                avg_voc_dur_chi = ('duration', np.mean)
            )
            
            metrics['cry_voc_chi_ph'] = vcm_agg.loc['Y', 'voc_chi_ph'] if 'Y' in vcm_agg.index else 0
            metrics['cry_voc_dur_chi_ph'] = vcm_agg.loc['Y', 'voc_dur_chi_ph'] if 'Y' in vcm_agg.index else 0

            if 'Y' in vcm_agg.index:
                metrics['avg_cry_voc_dur_chi'] = vcm_agg.loc['Y', 'avg_voc_dur_chi']

            metrics['can_voc_chi_ph'] = vcm_agg.loc['C', 'voc_chi_ph'] if 'C' in vcm_agg.index else 0
            metrics['can_voc_dur_chi_ph'] = vcm_agg.loc['C', 'voc_dur_chi_ph'] if 'C' in vcm_agg.index else 0

            if 'C' in vcm_agg.index:
                metrics['avg_can_voc_dur_chi'] = vcm_agg.loc['C', 'avg_voc_dur_chi']

            metrics['non_can_voc_chi_ph'] = vcm_agg.loc['N', 'voc_chi_ph'] if 'N' in vcm_agg.index else 0
            metrics['non_can_voc_dur_chi_ph'] = vcm_agg.loc['N', 'voc_dur_chi_ph'] if 'N' in vcm_agg.index else 0
            
            if 'N' in vcm_agg.index:        
                metrics['avg_non_can_voc_dur_chi'] = vcm_agg.loc['N', 'avg_voc_dur_chi']

            speech_voc = metrics['can_voc_chi_ph'] + metrics['non_can_voc_chi_ph']
            speech_dur = metrics['can_voc_dur_chi_ph'] + metrics['non_can_voc_dur_chi_ph']

            cry_voc = metrics['cry_voc_chi_ph']
            cry_dur = metrics['cry_voc_dur_chi_ph']

            if speech_voc+cry_voc:
                metrics['lp_n'] = speech_voc/(speech_voc+cry_voc)
                metrics['cp_n'] = metrics['can_voc_chi_ph']/speech_voc

                metrics['lp_dur'] = speech_dur/(speech_dur+cry_dur)
                metrics['cp_dur'] = metrics['can_voc_dur_chi_ph']/speech_dur

        metrics[self.by] = unit
        metrics['child_id'] = segments['child_id'].iloc[0]
        metrics['duration'] = unit_duration

        return metrics

    def extract(self):
        recordings = self.get_recordings()

        if self.threads == 1:
            self.metrics = pd.DataFrame([self._process_unit(unit) for unit in recordings[self.by].unique()])
        else:
            with mp.Pool(processes = self.threads if self.threads >= 1 else mp.cpu_count()) as pool:
                self.metrics = pd.DataFrame(pool.map(self._process_unit, recordings[self.by].unique()))

        self.metrics.set_index(self.by, inplace = True)
        return self.metrics
        

    @staticmethod
    def add_parser(subparsers, subcommand):
        parser = subparsers.add_parser(subcommand, help = 'LENA metrics')
        parser.add_argument('--vtc', help = 'vtc set', default = 'vtc')
        parser.add_argument('--alice', help = 'alice set', default = 'alice')
        parser.add_argument('--vcm', help = 'vcm set', default = 'vcm')
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
