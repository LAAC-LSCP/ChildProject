from collections import defaultdict
import datetime
import multiprocessing as mp
from numbers import Number
import numpy as np
import os
import pandas as pd
import pympi
import shutil
import sys
import traceback

from .projects import ChildProject
from .tables import IndexTable, IndexColumn

class AnnotationManager:
    INDEX_COLUMNS = [
        IndexColumn(name = 'set', description = 'name of the annotation set (e.g. VTC, annotator1, etc.)', required = True),
        IndexColumn(name = 'recording_filename', description = 'recording filename as specified in the recordings index', required = True),
        IndexColumn(name = 'time_seek', description = 'reference time in seconds, e.g: 3600, or 3600.500. All times expressed in the annotations are relative to this time.', regex = r"[0-9]{1,}(\.[0-9]{3})?", required = True),
        IndexColumn(name = 'range_onset', description = 'covered range start time in seconds, measured since `time_seek`', regex = r"[0-9]{1,}(\.[0-9]{3})?", required = True),
        IndexColumn(name = 'range_offset', description = 'covered range end time in seconds, measured since `time_seek`', regex = r"[0-9]{1,}(\.[0-9]{3})?", required = True),
        IndexColumn(name = 'raw_filename', description = 'annotation input filename location (relative to raw_annotations/)', filename = True, required = True),
        IndexColumn(name = 'format', description = 'input annotation format', regex = r"(TextGrid|eaf|vtc_rttm)", required = True),
        IndexColumn(name = 'filter', description = 'source file to filter in (for rttm only)', required = False),
        IndexColumn(name = 'annotation_filename', description = 'output formatted annotation location (automatic column, don\'t specify)', filename = True, required = False, generated = True),
        IndexColumn(name = 'imported_at', description = 'importation date (automatic column, don\'t specify)', datetime = "%Y-%m-%d %H:%M:%S", required = False, generated = True),
        IndexColumn(name = 'error', description = 'error message in case the annotation could not be imported', required = False, generated = True)
    ]

    SEGMENTS_COLUMNS = [
        IndexColumn(name = 'annotation_file', description = 'raw annotation path relative to /raw_annotations/', required = True),
        IndexColumn(name = 'segment_onset', description = 'segment start time in seconds', regex = r"(\d+(\.\d+)?)", required = True),
        IndexColumn(name = 'segment_offset', description = 'segment end time in seconds', regex = r"(\d+(\.\d+)?)", required = True),
        IndexColumn(name = 'speaker_id', description = 'identity of speaker in the annotation', required = True),
        IndexColumn(name = 'speaker_type', description = 'class of speaker (FEM, MAL, CHI, OCH)', regex = r"(FEM|MAL|CHI|OCH|SPEECH|NA)", required = True),
        IndexColumn(name = 'ling_type', description = '1 if the vocalization contains at least a vowel (ie canonical or non-canonical), 0 if crying or laughing', regex = r"(1|0|NA)", required = True),
        IndexColumn(name = 'vcm_type', description = 'vocal maturity defined as: C (canonical), N (non-canonical), Y (crying) L (laughing), J (junk)', regex = r"(C|N|Y|L|J|NA)", required = True),
        IndexColumn(name = 'lex_type', description = 'W if meaningful, 0 otherwise', regex = r"(W|0|NA)", required = True),
        IndexColumn(name = 'mwu_type', description = 'M if multiword, 1 if single word -- only filled if lex_type==W',regex = r"(M|1|NA)", required = True),
        IndexColumn(name = 'addresseee', description = 'T if target-child-directed, C if other-child-directed, A if adult-directed, U if uncertain or other', regex = r"(T|C|A|U|NA)", required = True),
        IndexColumn(name = 'transcription', description = 'orthographic transcription of the speach', required = True)
    ]

    SPEAKER_ID_TO_TYPE = {
        'C1': 'OCH',
        'C2': 'OCH',
        'CHI': 'CHI',
        'CHI*': 'CHI',
        'FA0': 'FEM',
        'FA1': 'FEM',
        'FA2': 'FEM',
        'FA3': 'FEM',
        'FA4': 'FEM',
        'FA5': 'FEM',
        'FA6': 'FEM',
        'FA7': 'FEM',
        'FA8': 'FEM',
        'FC1': 'OCH',
        'FC2': 'OCH',
        'FC3': 'OCH',
        'MA0': 'MAL',
        'MA1': 'MAL',
        'MA2': 'MAL',
        'MA3': 'MAL',
        'MA4': 'MAL',
        'MA5': 'MAL',
        'MC1': 'OCH',
        'MC2': 'OCH',
        'MC3': 'OCH',
        'MC4': 'OCH',
        'MC5': 'OCH',
        'MI1': 'OCH',
        'MOT*': 'FEM',
        'OC0': 'OCH',
        'UC1': 'OCH',
        'UC2': 'OCH',
        'UC3': 'OCH',
        'UC4': 'OCH',
        'UC5': 'OCH',
        'UC6': 'OCH'
    }

    VTC_SPEAKER_TYPE_TRANSLATION = defaultdict(lambda: 'NA', {
        'CHI': 'OCH',
        'KCHI': 'CHI',
        'FEM': 'FEM',
        'MAL':'MAL',
        'SPEECH': 'SPEECH'
    })


    def __init__(self, project):
        self.project = project
        self.annotations = None
        self.errors = []

        if not isinstance(project, ChildProject):
            raise ValueError('project should derive from ChildProject')

        project.read()

        index_path = os.path.join(self.project.path, 'metadata/annotations.csv')
        if not os.path.exists(index_path):
            open(index_path, 'w+').write(','.join([c.name for c in self.INDEX_COLUMNS]))

        errors, warnings = self.read()

    def read(self):
        table = IndexTable('input', path = os.path.join(self.project.path, 'metadata/annotations.csv'), columns = self.INDEX_COLUMNS)
        self.annotations = table.read()
        errors, warnings = table.validate()
        return errors, warnings

    def validate(self):
        errors, warnings = [], []

        for annotation in self.annotations.to_dict(orient = 'records'):
            segments = IndexTable(
                'segments',
                path = os.path.join(self.project.path, 'annotations', annotation['annotation_filename']),
                columns = self.SEGMENTS_COLUMNS
            )

            segments.read()
            res = segments.validate()
            errors += res[0]
            warnings += res[1]

        return errors, warnings
        

    def load_textgrid(self, filename):
        path = os.path.join(self.project.path, 'raw_annotations', filename)
        textgrid = pympi.Praat.TextGrid(path)

        def ling_type(s):
            s = str(s)

            a, b = ('0' in s, '1' in s)
            if a^b:
                return '0' if a else '1' 
            else:
                return 'NA'

        segments = []
        for tier in textgrid.tiers:
            for interval in tier.intervals:
                tier_name = tier.name.strip()

                if tier_name == 'Autre':
                    continue

                if interval[2] == "":
                    continue

                segment = {
                    'segment_onset': float(interval[0]),
                    'segment_offset': float(interval[1]),
                    'speaker_id': tier_name,
                    'ling_type': ling_type(interval[2]),
                    'speaker_type': self.SPEAKER_ID_TO_TYPE[tier_name] if tier_name in self.SPEAKER_ID_TO_TYPE else 'NA',
                    'vcm_type': 'NA',
                    'lex_type': 'NA',
                    'mwu_type': 'NA',
                    'addresseee': 'NA',
                    'transcription': 'NA'
                }

                segments.append(segment)

        return pd.DataFrame(segments)

    def load_eaf(self, filename):
        path = os.path.join(self.project.path, 'raw_annotations', filename)
        eaf = pympi.Elan.Eaf(path)

        segments = {}
        
        for tier_name in eaf.tiers:
            annotations = eaf.tiers[tier_name][0]

            if tier_name not in self.SPEAKER_ID_TO_TYPE and len(annotations) > 0:
                print("warning: unknown tier '{}' will be ignored in '{}'".format(tier_name, filename))
                continue

            for aid in annotations:
                (start_ts, end_ts, value, svg_ref) = annotations[aid]
                (start_t, end_t) = (eaf.timeslots[start_ts], eaf.timeslots[end_ts])

                segment = {
                    'segment_onset': start_t/1000,
                    'segment_offset': end_t/1000,
                    'speaker_id': tier_name,
                    'ling_type': 'NA',
                    'speaker_type': self.SPEAKER_ID_TO_TYPE[tier_name] if tier_name in self.SPEAKER_ID_TO_TYPE else 'NA',
                    'vcm_type': 'NA',
                    'lex_type': 'NA',
                    'mwu_type': 'NA',
                    'addresseee': 'NA',
                    'transcription': value if value != '0' else '0.'
                }

                segments[aid] = segment

        for tier_name in eaf.tiers:
            if '@' in tier_name:
                label, ref = tier_name.split('@')
            else:
                label, ref = tier_name, None

            reference_annotations = eaf.tiers[tier_name][1]

            if ref not in self.SPEAKER_ID_TO_TYPE:
                continue

            for aid in reference_annotations:
                (ann, value, prev, svg) = reference_annotations[aid]

                ann = aid
                parentTier = eaf.tiers[eaf.annotations[ann]]
                while 'PARENT_REF' in parentTier[2] and parentTier[2]['PARENT_REF'] and len(parentTier[2]) > 0:
                    ann = parentTier[1][ann][0]
                    parentTier = eaf.tiers[eaf.annotations[ann]]

                if ann not in segments:
                    print("warning: annotation '{}' not found in segments for '{}'".format(ann, filename))
                    continue
                
                segment = segments[ann]

                if label == 'lex':
                    segment['lex_type'] = value
                elif label == 'mwu':
                    segment['mwu_type'] = value
                elif label == 'xds':
                    segment['addresseee'] = value
                elif label == 'vcm':
                    segment['vcm_type'] = value

        return pd.DataFrame(segments.values())

    def load_vtc_rttm(self, filename, source_file = None):
        path = os.path.join(self.project.path, 'raw_annotations', filename)
        rttm = pd.read_csv(
            path,
            sep = " ",
            names = ['type', 'file', 'chnl', 'tbeg', 'tdur', 'ortho', 'stype', 'name', 'conf', 'unk']
        )

        df = rttm
        df['segment_onset'] = df['tbeg'].astype(float)
        df['segment_offset'] = (df['tbeg']+df['tdur']).astype(float)
        df['speaker_id'] = 'NA'
        df['ling_type'] = 'NA'
        df['speaker_type'] = df['name'].map(self.VTC_SPEAKER_TYPE_TRANSLATION)
        df['vcm_type'] = 'NA'
        df['lex_type'] = 'NA'
        df['mwu_type'] = 'NA'
        df['addresseee'] = 'NA'
        df['transcription'] = 'NA'  

        if source_file:
            df = df[df['file'] == source_file]

        df.drop(['type', 'file', 'chnl', 'tbeg', 'tdur', 'ortho', 'stype', 'name', 'conf', 'unk'], axis = 1, inplace = True)

        return df

    def import_annotation(self, annotation):
        source_recording = os.path.splitext(annotation['recording_filename'])[0]
        output_filename = "{}/{}_{}_{}.csv".format(annotation['set'], source_recording, annotation['time_seek'], annotation['range_onset'])

        raw_filename = annotation['raw_filename']
        annotation_format = annotation['format']

        df = None
        try:
            if annotation_format == 'TextGrid':
                df = self.load_textgrid(raw_filename)
            elif annotation_format == 'eaf':
                df = self.load_eaf(raw_filename)
            elif annotation_format == 'vtc_rttm':
                filter = annotation['filter'] if 'filter' in annotation and not pd.isnull(annotation['filter']) else None
                df = self.load_vtc_rttm(raw_filename, source_file = filter)
            else:
                raise ValueError("file format '{}' unknown for '{}'".format(annotation_format, raw_filename))
        except:
            annotation['error'] = traceback.format_exc()
            print("an error occured while processing '{}'".format(raw_filename), file = sys.stderr)
            print(traceback.format_exc(), file = sys.stderr)

        if df is None or not isinstance(df, pd.DataFrame):
            return annotation

        if not df.shape[1]:
            df = pd.DataFrame(columns = [c.name for c in self.SEGMENTS_COLUMNS])
        
        df['annotation_file'] = raw_filename
        df['segment_onset'] = df['segment_onset'].astype(float)
        df['segment_offset'] = df['segment_offset'].astype(float)

        if isinstance(annotation['range_onset'], Number)\
            and isinstance(annotation['range_offset'], Number)\
            and (annotation['range_offset'] - annotation['range_onset']) > 0.001:

            df = self.clip_segments(df, annotation['range_onset'], annotation['range_offset'])

        df.sort_values(['segment_onset', 'segment_offset', 'speaker_id', 'speaker_type'], inplace = True)

        os.makedirs(os.path.dirname(os.path.join(self.project.path, 'annotations', output_filename)), exist_ok = True)
        df.to_csv(os.path.join(self.project.path, 'annotations', output_filename), index = False)

        annotation['annotation_filename'] = output_filename
        annotation['imported_at'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return annotation

    def import_annotations(self, input):
        missing_recordings = input[~input['recording_filename'].isin(self.project.recordings['filename'].tolist())]
        missing_recordings = missing_recordings['recording_filename'].tolist()

        if len(missing_recordings) > 0:
            raise ValueError("cannot import annotations. the following recordings are incorrect:\n{}".format("\n".join(missing_recordings)))

        pool = mp.Pool(processes = mp.cpu_count())
        imported = pool.map(
            self.import_annotation,
            input.to_dict(orient = 'records')
        )

        imported = pd.DataFrame(imported)
        imported.drop(list(set(imported.columns)-set([c.name for c in self.INDEX_COLUMNS])), axis = 1, inplace = True)

        self.read()
        self.annotations = pd.concat([self.annotations, imported], sort = False)
        self.annotations.to_csv(os.path.join(self.project.path, 'metadata/annotations.csv'), index = False)

    def remove_set(self, annotation_set):
        self.read()

        try:
            shutil.rmtree(os.path.join(self.project.path, 'annotations', annotation_set))
        except:
            pass

        self.annotations = self.annotations[self.annotations['set'] != annotation_set]
        self.annotations.to_csv(os.path.join(self.project.path, 'metadata/annotations.csv'), index = False)

    def get_segments(self, annotations):
        annotations = annotations.dropna(subset = ['annotation_filename'])

        segments = pd.concat([
            pd.read_csv(os.path.join(self.project.path, 'annotations', f)).assign(annotation_filename = f)
            for f in annotations['annotation_filename'].tolist()
        ])

        return segments.merge(annotations, how = 'left', left_on = 'annotation_filename', right_on = 'annotation_filename')

    def clip_segments(self, segments, start, stop):
        segments['segment_onset'].clip(lower = start, upper = stop, inplace = True)
        segments['segment_offset'].clip(lower = start, upper = stop, inplace = True)

        segments = segments[~np.isclose(segments['segment_offset']-segments['segment_onset'], 0)]
        return segments

    def get_vc_stats(self, segments, turntakingthresh = 1):
        segments = segments.sort_values(['segment_onset', 'segment_offset'])
        segments = segments[segments['speaker_type'] != 'SPEECH']
        segments['duration'] = segments['segment_offset']-segments['segment_onset']
        segments['iti'] = segments['segment_onset'] - segments['segment_offset'].shift(1)
        segments['prev_speaker_type'] = segments['speaker_type'].shift(1)

        key_child_env = ['FEM', 'MAL', 'OCH']

        segments['turn'] = segments.apply(
            lambda row: (row['iti'] < turntakingthresh) and (
                (row['speaker_type'] == 'CHI' and row['prev_speaker_type'] in key_child_env) or
                (row['speaker_type'] in key_child_env and row['prev_speaker_type'] == 'CHI')
            ), axis = 1
        )

        segments['post_iti'] = segments['segment_onset'].shift(-1) - segments['segment_offset']
        segments['next_speaker_type'] = segments['speaker_type'].shift(-1)
        segments['cds'] = segments.apply(
            lambda row: row['duration'] if (
                (row['speaker_type'] == 'CHI' and row['prev_speaker_type'] in key_child_env and row['iti'] < turntakingthresh) or
                (row['speaker_type'] in key_child_env and row['prev_speaker_type'] == 'CHI' and row['iti'] < turntakingthresh) or
                (row['speaker_type'] == 'CHI' and row['next_speaker_type'] in key_child_env and row['post_iti'] < turntakingthresh) or
                (row['speaker_type'] in key_child_env and row['next_speaker_type'] == 'CHI' and row['post_iti'] < turntakingthresh)
            ) else 0, axis = 1
        )

        return segments.groupby('speaker_type').agg(
            cum_dur = ('duration', 'sum'),
            voc_count = ('duration', 'count'),
            turns = ('turn', 'sum'),
            cds_dur = ('cds', 'sum')
        )