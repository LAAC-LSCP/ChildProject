from collections import defaultdict
import datetime
from lxml import etree
import multiprocessing as mp
from numbers import Number
import numpy as np
import os
import pandas as pd
import pympi
import re
import shutil
import sys
import traceback

from .projects import ChildProject
from .tables import IndexTable, IndexColumn
from .utils import Segment, intersect_ranges

class AnnotationManager:
    INDEX_COLUMNS = [
        IndexColumn(name = 'set', description = 'name of the annotation set (e.g. VTC, annotator1, etc.)', required = True),
        IndexColumn(name = 'recording_filename', description = 'recording filename as specified in the recordings index', required = True),
        IndexColumn(name = 'time_seek', description = 'reference time in seconds, e.g: 3600, or 3600.500. All times expressed in the annotations are relative to this time.', regex = r"[0-9]{1,}(\.[0-9]{3})?", required = True),
        IndexColumn(name = 'range_onset', description = 'covered range start time in seconds, measured since `time_seek`', regex = r"(\d+(\.\d+)?)", required = True),
        IndexColumn(name = 'range_offset', description = 'covered range end time in seconds, measured since `time_seek`', regex = r"(\d+(\.\d+)?)", required = True),
        IndexColumn(name = 'raw_filename', description = 'annotation input filename location (relative to raw_annotations/)', filename = True, required = True),
        IndexColumn(name = 'format', description = 'input annotation format', choices = ['TextGrid', 'eaf', 'vtc_rttm', 'alice'], required = True),
        IndexColumn(name = 'filter', description = 'source file to filter in (for rttm and alice only)', required = False),
        IndexColumn(name = 'annotation_filename', description = 'output formatted annotation location (automatic column, don\'t specify)', filename = True, required = False, generated = True),
        IndexColumn(name = 'imported_at', description = 'importation date (automatic column, don\'t specify)', datetime = "%Y-%m-%d %H:%M:%S", required = False, generated = True),
        IndexColumn(name = 'error', description = 'error message in case the annotation could not be imported', required = False, generated = True)
    ]

    SEGMENTS_COLUMNS = [
        IndexColumn(name = 'annotation_file', description = 'raw annotation path relative to /raw_annotations/', required = True),
        IndexColumn(name = 'segment_onset', description = 'segment start time in seconds', regex = r"(\d+(\.\d+)?)", required = True),
        IndexColumn(name = 'segment_offset', description = 'segment end time in seconds', regex = r"(\d+(\.\d+)?)", required = True),
        IndexColumn(name = 'speaker_id', description = 'identity of speaker in the annotation', required = True),
        IndexColumn(name = 'speaker_type', description = 'class of speaker (FEM, MAL, CHI, OCH)', choices = ['FEM', 'MAL', 'CHI', 'OCH', 'SPEECH', 'NA'], required = True),
        IndexColumn(name = 'ling_type', description = '1 if the vocalization contains at least a vowel (ie canonical or non-canonical), 0 if crying or laughing', choices = ['1', '0', 'NA'], required = True),
        IndexColumn(name = 'vcm_type', description = 'vocal maturity defined as: C (canonical), N (non-canonical), Y (crying) L (laughing), J (junk)', choices = ['C', 'N', 'Y', 'L', 'J', 'NA'], required = True),
        IndexColumn(name = 'lex_type', description = 'W if meaningful, 0 otherwise', choices = ['W', '0', 'NA'], required = True),
        IndexColumn(name = 'mwu_type', description = 'M if multiword, 1 if single word -- only filled if lex_type==W', choices = ['M', '1', 'NA'], required = True),
        IndexColumn(name = 'addresseee', description = 'T if target-child-directed, C if other-child-directed, A if adult-directed, U if uncertain or other', choices = ['T', 'C', 'A', 'U', 'NA'], required = True),
        IndexColumn(name = 'transcription', description = 'orthographic transcription of the speach', required = True),
        IndexColumn(name = 'phonemes', description = 'amount of phonemes', regex = r'(\d+(\.\d+)?)'),
        IndexColumn(name = 'syllables', description = 'amount of syllables', regex = r'(\d+(\.\d+)?)'),
        IndexColumn(name = 'words', description = 'amount of words', regex = r'(\d+(\.\d+)?)'),
        IndexColumn(name = 'lena_block_type', description = 'whether regarded as part as a pause or a conversation by LENA', choices = ['pause', 'CM', 'CIC', 'CIOCX', 'CIOCAX', 'AMF', 'AICF', 'AIOCF', 'AIOCCXF', 'AMM', 'AICM', 'AIOCM', 'AIOCCXM', 'XM', 'XIOCC', 'XIOCA', 'XIC', 'XIOCAC']),
        IndexColumn(name = 'lena_block_number', description = 'number of the LENA pause/conversation the segment belongs to', regex = r"(\d+(\.\d+)?)"),
        IndexColumn(name = 'lena_conv_status', description = 'LENA conversation status', choices = ['BC', 'RC', 'EC']),
        IndexColumn(name = 'lena_response_count', description = 'LENA turn count within block', regex = r"(\d+(\.\d+)?)"),
        IndexColumn(name = 'lena_conv_floor_type', description = '(FI): Floor Initiation, (FH): Floor Holding', choices = ['FI', 'FH']),
        IndexColumn(name = 'lena_conv_turn_type', description = 'LENA turn type', choices = ['TIFI', 'TIMI', 'TIFR', 'TIMR', 'TIFE', 'TIME', 'NT']),
        IndexColumn(name = 'utterances_count', description = 'utterances count', regex = r"(\d+(\.\d+)?)"),
        IndexColumn(name = 'utterances_length', description = 'utterances length', regex = r"(\d+(\.\d+)?)"),
        IndexColumn(name = 'average_db', description = 'average dB level', regex = r"(?:-)(\d+(\.\d+)?)"),
        IndexColumn(name = 'peak_db', description = 'peak dB level', regex = r"(?:-)(\d+(\.\d+)?)"),
        IndexColumn(name = 'utterances', description = 'LENA utterances details (json)')
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

    LENA_SPEAKER_TYPE_TRANSLATION = {
        'CHN': 'CHI',
        'CXN': 'OCH',
        'FAN': 'FEM',
        'MAN': 'MAL',
        'OLN': 'OLN',
        'TVN': 'TVN',
        'NON': 'NON',
        'SIL': 'SIL',
        'FUZ': 'FUZ',
        'TVF': 'TVF',
        'CXF': 'CXF',
        'NOF': 'NON',
        'OLF': 'OLN',
        'CHF': 'CHF',
        'MAF': 'MAF',
        'FAF': 'FEF'
    }


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

        self.errors, self.warnings = self.read()

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
                    'transcription': 'NA',
                    'phonemes': 'NA',
                    'syllables': 'NA',
                    'words':'NA'
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
                    'transcription': value if value != '0' else '0.',
                    'phonemes': 'NA',
                    'syllables': 'NA',
                    'words':'NA'
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

    def load_its(self, filename):
        path = os.path.join(self.project.path, 'raw_annotations', filename)
        xml = etree.parse(path)

        recordings = xml.xpath('/ITS/ProcessingUnit/Recording')
        timestamp_pattern = re.compile(r"^P(?:T?)(\d+(\.\d+)?)S$")

        def extract_from_regex(pattern, subject):
            match = pattern.search(subject)
            return match.group(1) if match else ''

        segments = []

        for recording in recordings:
            segs = recording.xpath('//Segment')
            for seg in segs:
                parent = seg.getparent()

                lena_block_number = parent.get('num')
                lena_block_type = 'pause' if parent.tag.lower() == 'pause' else parent.get('type')

                if not seg.get('conversationInfo'):
                    conversation_info = ['NA'] * 7
                else:
                    conversation_info = seg.get('conversationInfo').split('|')[1:-1]
                
                lena_conv_status = conversation_info[0]
                lena_response_count = conversation_info[3]
                lena_conv_turn_type = conversation_info[5]
                lena_conv_floor_type = conversation_info[6]

                onset = extract_from_regex(timestamp_pattern, seg.get('startTime'))
                offset = extract_from_regex(timestamp_pattern, seg.get('endTime'))

                words = 0
                for attr in ['femaleAdultWordCnt', 'maleAdultWordCnt']:
                    words += float(seg.get(attr, 0))

                utterances_count = 0
                for attr in ['femaleAdultUttCnt', 'maleAdultUttCnt', 'childUttCnt']:
                    utterances_count += float(seg.get(attr, 0))

                utterances_length = 0
                for attr in ['femaleAdultUttLen', 'maleAdultUttLen', 'childUttLen']:
                    utterances_length += float(extract_from_regex(timestamp_pattern, seg.get(attr, 'P0S')))

                average_db = seg.get('average_dB', 0)
                peak_db = seg.get('peak_dB', 0)

                utterances = seg.xpath('./UTT')
                utterances = [utt.attrib for utt in utterances]

                segments.append({
                    'segment_onset': onset,
                    'segment_offset': offset,
                    'speaker_id': 'NA',
                    'ling_type': 'NA',
                    'speaker_type': self.LENA_SPEAKER_TYPE_TRANSLATION[seg.get('spkr')],
                    'vcm_type': 'NA',
                    'lex_type': 'NA',
                    'mwu_type': 'NA',
                    'addresseee': 'NA',
                    'transcription': 'NA',
                    'phonemes': 'NA',
                    'syllables': 'NA',
                    'words': words,
                    'lena_block_number': lena_block_number,
                    'lena_block_type': lena_block_type,
                    'lena_conv_status': lena_conv_status,
                    'lena_response_count': lena_response_count,
                    'lena_conv_turn_type': lena_conv_turn_type,
                    'lena_conv_floor_type': lena_conv_floor_type,
                    'utterances_count': utterances_count,
                    'utterances_length': utterances_length,
                    'average_db': average_db,
                    'peak_db': peak_db,
                    'utterances': utterances
                })
                
        df = pd.DataFrame(segments)
        
        return df

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
        df['phonemes'] = 'NA'
        df['syllables'] = 'NA'
        df['words'] = 'NA'

        if source_file:
            df = df[df['file'] == source_file]

        df.drop(['type', 'file', 'chnl', 'tbeg', 'tdur', 'ortho', 'stype', 'name', 'conf', 'unk'], axis = 1, inplace = True)

        return df

    def load_alice(self, filename, source_file = None):
        path = os.path.join(self.project.path, 'raw_annotations', filename)
        df = pd.read_csv(
            path,
            sep = r"\s",
            names = ['file', 'phonemes', 'syllables', 'words']
        )
        df['speaker_id'] = 'NA'
        df['ling_type'] = 'NA'
        df['speaker_type'] = 'NA'
        df['vcm_type'] = 'NA'
        df['lex_type'] = 'NA'
        df['mwu_type'] = 'NA'
        df['addresseee'] = 'NA'
        df['transcription'] = 'NA'

        if source_file:
            df = df[df['file'].str.contains(source_file)]

        matches = df['file'].str.extract(r"^(.*)_(?:0+)?([0-9]{1,})_(?:0+)?([0-9]{1,})\.wav$")
        df['filename'] = matches[0]
        df['segment_onset'] = matches[1].astype(float)/10000
        df['segment_offset'] = matches[2].astype(float)/10000
        
        df.drop(columns = ['filename', 'file'], inplace = True)

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
            elif annotation_format == 'alice':
                filter = annotation['filter'] if 'filter' in annotation and not pd.isnull(annotation['filter']) else None
                df = self.load_alice(raw_filename, source_file = filter)
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

    def import_annotations(self, input, threads = -1):
        missing_recordings = input[~input['recording_filename'].isin(self.project.recordings['filename'].tolist())]
        missing_recordings = missing_recordings['recording_filename'].tolist()

        if len(missing_recordings) > 0:
            raise ValueError("cannot import annotations. the following recordings are incorrect:\n{}".format("\n".join(missing_recordings)))

        pool = mp.Pool(processes = threads if threads > 0 else mp.cpu_count())
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

    def merge_sets(self, left_set, right_set, left_columns, right_columns, output_set, columns = {}):
        assert left_set != right_set, "sets must differ"
        assert not (set(left_columns) & set (right_columns)), "left_columns and right_columns must be disjoint"

        union = set(left_columns) | set (right_columns)
        all_columns = set([c.name for c in self.SEGMENTS_COLUMNS]) - set(['annotation_file', 'segment_onset', 'segment_offset'])
        required_columns = set([c.name for c in self.SEGMENTS_COLUMNS if c.required]) - set(['annotation_file', 'segment_onset', 'segment_offset'])
        assert union.issubset(all_columns), "left_columns and right_columns have unexpected values"
        assert required_columns.issubset(union), "left_columns and right_columns have missing values"


        left_annotations = self.annotations[self.annotations['set'] == left_set]
        right_annotations = self.annotations[self.annotations['set'] == right_set]

        left_annotations, right_annotations = self.intersection(left_annotations, right_annotations)
        left_annotations = left_annotations.reset_index(drop = True).rename_axis('interval').reset_index()
        right_annotations = right_annotations.reset_index(drop = True).rename_axis('interval').reset_index()

        annotations = left_annotations.copy()
        annotations['format'] = ''
        annotations['annotation_filename'] = annotations.apply(
            lambda annotation: "{}/{}_{}_{}.csv".format(
                output_set,
                os.path.splitext(annotation['recording_filename'])[0],
                annotation['time_seek'],
                annotation['range_onset']
            )
        , axis = 1)

        for key in columns:
            annotations[key] = columns[key]

        annotations['set'] = output_set

        left_segments = self.get_segments(left_annotations)
        left_segments['segment_onset'] = left_segments['segment_onset'] + left_segments['time_seek']
        left_segments['segment_offset'] = left_segments['segment_offset'] + left_segments['time_seek']

        right_segments = self.get_segments(right_annotations)
        right_segments['segment_onset'] = right_segments['segment_onset'] + right_segments['time_seek']
        right_segments['segment_offset'] = right_segments['segment_offset'] + right_segments['time_seek']

        def timestamp_to_int(f):
            return int(round(f*10000))

        def int_to_timestamp(i):
            return i/10000

        left_segments['segment_onset'] = left_segments['segment_onset'].apply(timestamp_to_int)
        left_segments['segment_offset'] = left_segments['segment_offset'].apply(timestamp_to_int)
        right_segments['segment_onset'] = right_segments['segment_onset'].apply(timestamp_to_int)
        right_segments['segment_offset'] = right_segments['segment_offset'].apply(timestamp_to_int)

        merge_columns = ['interval', 'segment_onset', 'segment_offset']

        output_segments = left_segments[merge_columns + left_columns + ['annotation_file', 'time_seek']].merge(
            right_segments[merge_columns + right_columns + ['annotation_file']],
            how = 'outer',
            left_on = merge_columns,
            right_on = merge_columns
        )
        output_segments['segment_onset'] = output_segments['segment_onset'].apply(int_to_timestamp)
        output_segments['segment_offset'] = output_segments['segment_offset'].apply(int_to_timestamp)

        output_segments['segment_onset'] = output_segments['segment_onset'] - output_segments['time_seek']
        output_segments['segment_offset'] = output_segments['segment_offset'] - output_segments['time_seek']

        output_segments['annotation_file'] = output_segments['annotation_file_x'] + ',' + output_segments['annotation_file_y']

        annotations.drop(columns = 'raw_filename', inplace = True)
        annotations = annotations.merge(
            output_segments[['interval', 'annotation_file']].dropna().drop_duplicates(),
            how = 'left',
            left_on = 'interval',
            right_on = 'interval'
        )
        annotations.rename(columns = {'annotation_file': 'raw_filename'}, inplace = True)
        annotations['generated_at'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        output_segments['annotation_file'] = output_segments['annotation_file_x'].fillna('') + ',' + output_segments['annotation_file_y'].fillna('')
        output_segments.drop(columns = ['annotation_file_x', 'annotation_file_y', 'time_seek'], inplace = True)

        output_segments.fillna('NA', inplace = True)

        for interval, segments in output_segments.groupby('interval'):
            annotation_filename = annotations[annotations['interval'] == interval]['annotation_filename'].tolist()[0]
            os.makedirs(os.path.dirname(os.path.join(self.project.path, 'annotations', annotation_filename)), exist_ok = True)

            segments.drop(columns = list(set(segments.columns)-set([c.name for c in self.SEGMENTS_COLUMNS])), inplace = True)
            segments.to_csv(
                os.path.join(self.project.path, 'annotations', annotation_filename),
                index = False
            )

        annotations.drop(columns = list(set(annotations.columns)-set([c.name for c in self.INDEX_COLUMNS])), inplace = True)
        
        self.read()
        self.annotations = pd.concat([self.annotations, annotations], sort = False)
        self.annotations.to_csv(os.path.join(self.project.path, 'metadata/annotations.csv'), index = False)

    def get_segments(self, annotations):
        annotations = annotations.dropna(subset = ['annotation_filename'])

        segments = pd.concat([
            pd.read_csv(os.path.join(self.project.path, 'annotations', f)).assign(annotation_filename = f)
            for f in annotations['annotation_filename'].tolist()
        ])

        return segments.merge(annotations, how = 'left', left_on = 'annotation_filename', right_on = 'annotation_filename')

    def intersection(self, left, right):
        recordings = set(left['recording_filename'].unique()) & set(right['recording_filename'].unique())
        recordings = list(recordings)

        a_stack = []
        b_stack = []

        for recording in recordings:
            a = left[left['recording_filename'] == recording]
            b = right[right['recording_filename'] == recording]

            for bound in ('onset', 'offset'):
                a['abs_range_' + bound] = a['range_' + bound] + a['time_seek']
                b['abs_range_' + bound] = b['range_' + bound] + b['time_seek']

            a_ranges = a[['abs_range_onset', 'abs_range_offset']].sort_values(['abs_range_onset', 'abs_range_offset']).values.tolist()
            b_ranges = b[['abs_range_onset', 'abs_range_offset']].sort_values(['abs_range_onset', 'abs_range_offset']).values.tolist()

            segments = list(intersect_ranges(
                (Segment(onset, offset) for (onset, offset) in a_ranges),
                (Segment(onset, offset) for (onset, offset) in b_ranges)
            ))

            a_out = []
            b_out = []

            for segment in segments:
                a_row = a[(a['abs_range_onset'] <= segment.start) & (a['abs_range_offset'] >= segment.stop)].to_dict(orient = 'records')[0]
                a_row['abs_range_onset'] = segment.start
                a_row['abs_range_offset'] = segment.stop
                a_out.append(a_row)

                b_row = b[(b['abs_range_onset'] <= segment.start) & (b['abs_range_offset'] >= segment.stop)].to_dict(orient = 'records')[0]
                b_row['abs_range_onset'] = segment.start
                b_row['abs_range_offset'] = segment.stop
                b_out.append(b_row)

            if not a_out or not b_out:
                continue

            a_out = pd.DataFrame(a_out)
            b_out = pd.DataFrame(b_out)

            for bound in ('onset', 'offset'):
                a_out['range_' + bound] = a_out['abs_range_' + bound] - a_out['time_seek']
                b_out['range_' + bound] = b_out['abs_range_' + bound] - b_out['time_seek']

            a_out.drop(['abs_range_onset', 'abs_range_offset'], axis = 1, inplace = True)
            b_out.drop(['abs_range_onset', 'abs_range_offset'], axis = 1, inplace = True)

            a_stack.append(a_out)
            b_stack.append(b_out)

        return pd.concat(a_stack), pd.concat(b_stack)

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