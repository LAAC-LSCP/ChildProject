import datetime
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from functools import reduce, partial
from shutil import move, rmtree
import sys
import traceback
from typing import Callable

from . import __version__
from .projects import ChildProject
from .tables import IndexTable, IndexColumn
from .utils import Segment, intersect_ranges

class AnnotationManager:
    """Manage annotations of a dataset. 

    Attributes:
        :param project: :class:`ChildProject.projects.ChildProject` instance of the target dataset.
        :type project: :class:`ChildProject.projects.ChildProject`
    """
    INDEX_COLUMNS = [
        IndexColumn(name = 'set', description = 'name of the annotation set (e.g. VTC, annotator1, etc.)', required = True),
        IndexColumn(name = 'recording_filename', description = 'recording filename as specified in the recordings index', required = True),
        IndexColumn(name = 'time_seek', description = 'reference time in milliseconds, e.g: 3600000. All times expressed in the annotations are relative to this time.', regex = r"(\-?)([0-9]+)", required = True),
        IndexColumn(name = 'range_onset', description = 'covered range start time in milliseconds, measured since `time_seek`', regex = r"([0-9]+)", required = True),
        IndexColumn(name = 'range_offset', description = 'covered range end time in milliseconds, measured since `time_seek`', regex = r"([0-9]+)", required = True),
        IndexColumn(name = 'raw_filename', description = 'annotation input filename location, relative to `annotations/<set>/raw`', filename = True, required = True),
        IndexColumn(name = 'format', description = 'input annotation format', choices = ['TextGrid', 'eaf', 'vtc_rttm', 'vcm_rttm', 'alice', 'its', 'chat'], required = False),
        IndexColumn(name = 'filter', description = 'source file to filter in (for rttm and alice only)', required = False),
        IndexColumn(name = 'annotation_filename', description = 'output formatted annotation location, relative to `annotations/<set>/converted (automatic column, don\'t specify)', filename = True, required = False, generated = True),
        IndexColumn(name = 'imported_at', description = 'importation date (automatic column, don\'t specify)', datetime = "%Y-%m-%d %H:%M:%S", required = False, generated = True),
        IndexColumn(name = 'package_version', description = 'version of the package used when the importation was performed', regex = r"[0-9]+\.[0-9]+\.[0-9]+", required = False, generated = True),
        IndexColumn(name = 'error', description = 'error message in case the annotation could not be imported', required = False, generated = True)
    ]

    SEGMENTS_COLUMNS = [
        IndexColumn(name = 'raw_filename', description = 'raw annotation path relative, relative to `annotations/<set>/raw`', required = True),
        IndexColumn(name = 'segment_onset', description = 'segment start time in milliseconds', regex = r"([0-9]+)", required = True),
        IndexColumn(name = 'segment_offset', description = 'segment end time in milliseconds', regex = r"([0-9]+)", required = True),
        IndexColumn(name = 'speaker_id', description = 'identity of speaker in the annotation'),
        IndexColumn(name = 'speaker_type', description = 'class of speaker (FEM, MAL, CHI, OCH)', choices = ['FEM', 'MAL', 'CHI', 'OCH', 'SPEECH'] + ['TVN', 'TVF', 'FUZ', 'FEF', 'MAF', 'SIL', 'CXF', 'NON', 'OLN', 'OLF', 'CHF', 'NA']),
        IndexColumn(name = 'ling_type', description = '1 if the vocalization contains at least a vowel (ie canonical or non-canonical), 0 if crying or laughing', choices = ['1', '0', 'NA']),
        IndexColumn(name = 'vcm_type', description = 'vocal maturity defined as: C (canonical), N (non-canonical), Y (crying) L (laughing), J (junk)', choices = ['C', 'N', 'Y', 'L', 'J', 'NA']),
        IndexColumn(name = 'lex_type', description = 'W if meaningful, 0 otherwise', choices = ['W', '0', 'NA']),
        IndexColumn(name = 'mwu_type', description = 'M if multiword, 1 if single word -- only filled if lex_type==W', choices = ['M', '1', 'NA']),
        IndexColumn(name = 'addressee', description = 'T if target-child-directed, C if other-child-directed, A if adult-directed, U if uncertain or other. Multiple values should be sorted and separated by commas', choices = ['T', 'C', 'A', 'U', 'NA']),
        IndexColumn(name = 'transcription', description = 'orthographic transcription of the speach'),
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
        IndexColumn(name = 'utterances_length', description = 'utterances length', regex = r"([0-9]+)"),
        IndexColumn(name = 'non_speech_length', description = 'non-speech length', regex = r"([0-9]+)"),
        IndexColumn(name = 'average_db', description = 'average dB level', regex = r"(\-?)(\d+(\.\d+)?)"),
        IndexColumn(name = 'peak_db', description = 'peak dB level', regex = r"(\-?)(\d+(\.\d+)?)"),
        IndexColumn(name = 'child_cry_vfx_len', description = 'childCryVfxLen', regex = r"([0-9]+)"),
        IndexColumn(name = 'utterances', description = 'LENA utterances details (json)'),
        IndexColumn(name = 'cries', description = 'cries (json)'),
        IndexColumn(name = 'vfxs', description = 'Vfx (json)')
    ]

    def __init__(self, project: ChildProject):
        """AnnotationManager constructor

        :param project: :class:`ChildProject` instance of the target dataset.
        :type project: :class:`ChildProject`
        """
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

    def read(self) -> tuple:
        table = IndexTable('input', path = os.path.join(self.project.path, 'metadata/annotations.csv'), columns = self.INDEX_COLUMNS)
        self.annotations = table.read()
        errors, warnings = table.validate()
        return errors, warnings

    def validate_annotation(self, annotation: dict) -> tuple:
        print("validating {}...".format(annotation['annotation_filename']))

        segments = IndexTable(
            'segments',
            path = os.path.join(self.project.path, 'annotations', annotation['set'], 'converted', str(annotation['annotation_filename'])),
            columns = self.SEGMENTS_COLUMNS
        )

        try:
            segments.read()
        except Exception as e:
            return [str(e)], []

        return segments.validate()

    def validate(self, annotations: pd.DataFrame = None, threads: int = -1) -> tuple:
        if not isinstance(annotations, pd.DataFrame):
            annotations = self.annotations

        errors, warnings = [], []

        pool = mp.Pool(processes = threads if threads > 0 else mp.cpu_count())
        res = pool.map(self.validate_annotation, annotations.to_dict(orient = 'records'))

        errors = reduce(lambda x,y: x+y[0], res, [])
        warnings = reduce(lambda x,y: x+y[1], res, [])

        return errors, warnings


    def _import_annotation(self, import_function: Callable[[str], pd.DataFrame], annotation: dict):
        """import and convert ``annotation``. This function should not be called outside of this class.

        :param import_function: If callable, ``import_function`` will be called to convert the input annotation into a dataframe. Otherwise, the conversion will be performed by a built-in function.
        :type import_function: Callable[[str], pd.DataFrame]
        :param annotation: input annotation dictionary (attributes defined according to :ref:`ChildProject.annotations.AnnotationManager.SEGMENTS_COLUMNS`)
        :type annotation: dict
        :return: output annotation dictionary (attributes defined according to :ref:`ChildProject.annotations.AnnotationManager.SEGMENTS_COLUMNS`)
        :rtype: dict
        """
        from .converters import TextGridConverter,EafConverter,VtcConverter,VcmConverter,ItsConverter,AliceConverter,ChatConverter

        source_recording = os.path.splitext(annotation['recording_filename'])[0]
        annotation_filename = "{}_{}_{}.csv".format(source_recording, annotation['time_seek'], annotation['range_onset'])
        output_filename = os.path.join('annotations', annotation['set'], 'converted', annotation_filename)

        path = os.path.join(self.project.path, 'annotations', annotation['set'], 'raw', annotation['raw_filename'])
        annotation_format = annotation['format']

        df = None
        filter = annotation['filter'] if 'filter' in annotation and not pd.isnull(annotation['filter']) else None

        try:
            if callable(import_function):
                df = import_function(path)
            elif annotation_format == 'TextGrid':
                df = TextGridConverter.convert(path)
            elif annotation_format == 'eaf':
                df = EafConverter.convert(path)
            elif annotation_format == 'vtc_rttm':
                df = VtcConverter.convert(path, source_file = filter)
            elif annotation_format == 'vcm_rttm':
                df = VcmConverter.convert(path, source_file = filter)
            elif annotation_format == 'its':
                df = ItsConverter.convert(path, recording_num = filter)
            elif annotation_format == 'alice':
                df = AliceConverter.convert(path, source_file = filter)
            elif annotation_format == 'chat':
                df = ChatConverter.convert(path)
            else:
                raise ValueError("file format '{}' unknown for '{}'".format(annotation_format, path))
        except:
            annotation['error'] = traceback.format_exc()
            print("an error occured while processing '{}'".format(path), file = sys.stderr)
            print(traceback.format_exc(), file = sys.stderr)

        if df is None or not isinstance(df, pd.DataFrame):
            return annotation

        if not df.shape[1]:
            df = pd.DataFrame(columns = [c.name for c in self.SEGMENTS_COLUMNS])
        
        df['raw_filename'] = annotation['raw_filename']
        df['segment_onset'] = df['segment_onset'].astype(int)
        df['segment_offset'] = df['segment_offset'].astype(int)

        annotation['range_onset'] = int(annotation['range_onset'])
        annotation['range_offset'] = int(annotation['range_offset'])

        df = self.clip_segments(df, annotation['range_onset'], annotation['range_offset'])

        sort_columns = ['segment_onset', 'segment_offset']
        if 'speaker_type' in df.columns:
            sort_columns.append('speaker_type')

        df.sort_values(sort_columns, inplace = True)

        os.makedirs(os.path.dirname(os.path.join(self.project.path, output_filename)), exist_ok = True)
        df.to_csv(os.path.join(self.project.path, output_filename), index = False)

        annotation['annotation_filename'] = annotation_filename
        annotation['imported_at'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        annotation['package_version'] = __version__

        return annotation

    def import_annotations(self, input: pd.DataFrame, threads: int = -1, import_function: Callable[[str], pd.DataFrame] = None) -> pd.DataFrame:
        """Import and convert annotations.

        :param input: dataframe of all annotations to import, as described in :ref:`format-input-annotations`.
        :type input: pd.DataFrame
        :param threads: If > 1, conversions will be run on ``threads`` threads, defaults to -1
        :type threads: int, optional
        :param import_function: If specified, the custom ``import_function`` function will be used to convert all ``input`` annotations, defaults to None
        :type import_function: Callable[[str], pd.DataFrame], optional
        :return: dataframe of imported annotations, as in :ref:`format-annotations`.
        :rtype: pd.DataFrame
        """
        
        missing_recordings = input[~input['recording_filename'].isin(self.project.recordings['recording_filename'].tolist())]
        missing_recordings = missing_recordings['recording_filename'].tolist()

        if len(missing_recordings) > 0:
            raise ValueError("cannot import annotations, because the following recordings are not referenced in the metadata:\n{}".format("\n".join(missing_recordings)))

        input['range_onset'] = input['range_onset'].astype(int)
        input['range_offset'] = input['range_offset'].astype(int)

        if 'chat' in input['format'].tolist():
            print('warning: chat does not support multithread importation; running on 1 thread')
            threads = 1

        if threads == 1:
            imported = input.apply(partial(self._import_annotation, import_function), axis = 1).to_dict(orient = 'records')
        else:
            pool = mp.Pool(processes = threads if threads > 0 else mp.cpu_count())
            imported = pool.map(
                partial(self._import_annotation, import_function),
                input.to_dict(orient = 'records')
            )

        imported = pd.DataFrame(imported)
        imported.drop(list(set(imported.columns)-{c.name for c in self.INDEX_COLUMNS}), axis = 1, inplace = True)

        self.read()
        self.annotations = pd.concat([self.annotations, imported], sort = False)
        self.annotations.to_csv(os.path.join(self.project.path, 'metadata/annotations.csv'), index = False)

        return imported

    def get_subsets(self, annotation_set: str, recursive: bool = False) -> list:
        """Retrieve the list of subsets belonging to a given set of annotations.

        :param annotation_set: input set
        :type annotation_set: str
        :param recursive: If True, get subsets recursively, defaults to False
        :type recursive: bool, optional
        :return: the list of subsets names
        :rtype: list
        """
        subsets = []

        path = os.path.join(self.project.path, 'annotations', annotation_set)
        candidates = list(set(os.listdir(path)) - {'raw', 'converted'})
        for candidate in candidates:
            subset = os.path.join(annotation_set, candidate)

            if not os.path.isdir(os.path.join(self.project.path, 'annotations', subset)):
                continue

            subsets.append(subset)

            if recursive:
                subsets.extend(self.get_subsets(subset))

        return subsets
            

    def remove_set(self, annotation_set: str, recursive: bool = False):
        """Remove a set of annotations, deleting every file and removing
        them from the index.

        :param annotation_set: set of annotations to remove
        :type annotation_set: str
        :param recursive: remove subsets as well, defaults to False
        :type recursive: bool, optional
        """
        self.read()

        subsets = []
        if recursive:
            subsets = self.get_subsets(annotation_set, recursive = False)

        for subset in subsets:
            self.remove_set(subset, recursive = recursive)

        path = os.path.join(self.project.path, 'annotations', annotation_set, 'converted')

        try:
            rmtree(path)
        except:
            print("could not delete '{}', as it does not exist (yet?)".format(path))
            pass

        self.annotations = self.annotations[self.annotations['set'] != annotation_set]
        self.annotations.to_csv(os.path.join(self.project.path, 'metadata/annotations.csv'), index = False)

    def rename_set(self, annotation_set: str, new_set: str, recursive: bool = False, ignore_errors: bool = False):
        """Rename a set of annotations, moving all related files
        and updating the index accordingly.

        :param annotation_set: name of the set to rename
        :type annotation_set: str
        :param new_set: new set name
        :type new_set: str
        :param recursive: rename subsets as well, defaults to False
        :type recursive: bool, optional
        :param ignore_errors: If True, keep going even if unindexed files are detected, defaults to False
        :type ignore_errors: bool, optional
        """
        self.read()

        annotation_set = annotation_set.rstrip('/').rstrip("\\")
        new_set = new_set.rstrip('/').rstrip("\\")

        current_path = os.path.join(self.project.path, 'annotations', annotation_set)
        new_path = os.path.join(self.project.path, 'annotations', new_set)

        if not os.path.exists(current_path):
            raise Exception("'{}' does not exists, aborting".format(current_path))

        if os.path.exists(new_path):
            raise Exception("'{}' already exists, aborting".format(new_path))

        if self.annotations[self.annotations['set'] == annotation_set].shape[0] == 0 and not ignore_errors and not recursive:
            raise Exception("set '{}' have no indexed annotation, aborting. use --ignore_errors to force")

        subsets = []
        if recursive:
            subsets = self.get_subsets(annotation_set, recursive = False)

        for subset in subsets:
            self.rename_set(
                annotation_set = subset,
                new_set = re.sub(r"^{}/".format(re.escape(annotation_set)), os.path.join(new_set, ''), subset),
                recursive = recursive,
                ignore_errors = ignore_errors
            )

        os.makedirs(new_path, exist_ok = True)

        if os.path.exists(os.path.join(current_path, 'raw')):
            move(os.path.join(current_path, 'raw'), os.path.join(new_path, 'raw'))

        if os.path.exists(os.path.join(current_path, 'converted')):
            move(os.path.join(current_path, 'converted'), os.path.join(new_path, 'converted'))

        self.annotations.loc[(self.annotations['set'] == annotation_set), 'set'] = new_set

        self.annotations.to_csv(os.path.join(self.project.path, 'metadata/annotations.csv'), index = False)

    def merge_annotations(self, left_columns, right_columns, columns, output_set, input):
        left_annotations = input['left_annotations']
        right_annotations = input['right_annotations']
        
        annotations = left_annotations.copy()
        annotations['format'] = ''
        annotations['annotation_filename'] = annotations.apply(
            lambda annotation: "{}_{}_{}.csv".format(
                os.path.splitext(annotation['recording_filename'])[0],
                annotation['time_seek'],
                annotation['range_onset']
            )
        , axis = 1)

        for key in columns:
            annotations[key] = columns[key]

        annotations['set'] = output_set

        left_annotation_files = [os.path.join(self.project.path, 'annotations', a['set'], 'converted', a['annotation_filename']) for a in left_annotations.to_dict(orient = 'records')]
        left_missing_annotations = [f for f in left_annotation_files if not os.path.exists(f)]

        right_annotation_files = [os.path.join(self.project.path, 'annotations', a['set'], 'converted', a['annotation_filename']) for a in right_annotations.to_dict(orient = 'records')]
        right_missing_annotations = [f for f in right_annotation_files if not os.path.exists(f)]

        if left_missing_annotations:
            raise Exception('the following annotations from the left set are missing: {}'.format(','.join(left_missing_annotations)))

        if right_missing_annotations:
            raise Exception('the following annotations from the right set are missing: {}'.format(','.join(right_missing_annotations)))

        left_segments = self.get_segments(left_annotations)
        left_segments['segment_onset'] = left_segments['segment_onset'] + left_segments['time_seek']
        left_segments['segment_offset'] = left_segments['segment_offset'] + left_segments['time_seek']

        right_segments = self.get_segments(right_annotations)
        right_segments['segment_onset'] = right_segments['segment_onset'] + right_segments['time_seek']
        right_segments['segment_offset'] = right_segments['segment_offset'] + right_segments['time_seek']

        merge_columns = ['interval', 'segment_onset', 'segment_offset']

        output_segments = left_segments[merge_columns + left_columns + ['raw_filename', 'time_seek']].merge(
            right_segments[merge_columns + right_columns + ['raw_filename']],
            how = 'outer',
            left_on = merge_columns,
            right_on = merge_columns
        )

        output_segments['segment_onset'] = (output_segments['segment_onset'] - output_segments['time_seek']).fillna(0).astype(int)
        output_segments['segment_offset'] = (output_segments['segment_offset'] - output_segments['time_seek']).fillna(0).astype(int)

        output_segments['raw_filename'] = output_segments['raw_filename_x'] + ',' + output_segments['raw_filename_y']

        annotations.drop(columns = 'raw_filename', inplace = True)
        annotations = annotations.merge(
            output_segments[['interval', 'raw_filename']].dropna().drop_duplicates(),
            how = 'left',
            left_on = 'interval',
            right_on = 'interval'
        )
        annotations.rename(columns = {'raw_filename': 'raw_filename'}, inplace = True)
        annotations['generated_at'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        output_segments['raw_filename'] = output_segments['raw_filename_x'].fillna('') + ',' + output_segments['raw_filename_y'].fillna('')
        output_segments.drop(columns = ['raw_filename_x', 'raw_filename_y', 'time_seek'], inplace = True)

        output_segments.fillna('NA', inplace = True)

        for annotation in annotations.to_dict(orient = 'records'):
            interval = annotation['interval']
            annotation_filename = annotation['annotation_filename']
            annotation_set = annotation['set']

            os.makedirs(os.path.dirname(os.path.join(self.project.path, 'annotations', annotation_set, 'converted', annotation_filename)), exist_ok = True)

            segments = output_segments[output_segments['interval'] == interval]
            segments.drop(columns = list(set(segments.columns)-{c.name for c in self.SEGMENTS_COLUMNS}), inplace = True)
            segments.to_csv(
                os.path.join(self.project.path, 'annotations', annotation_set, 'converted', annotation_filename),
                index = False
            )

        return annotations

    def merge_sets(self, left_set: str, right_set: str,
        left_columns: list, right_columns: list,
        output_set: str, columns: dict = {},
        threads = -1
    ):
        """Merge columns from ``left_set`` and ``right_set`` annotations, 
        for all matching segments, into a new set of annotations named
        ``output_set``.

        :param left_set: Left set of annotations.
        :type left_set: str
        :param right_set: Right set of annotations.
        :type right_set: str
        :param left_columns: Columns which values will be based on the left set.
        :type left_columns: list
        :param right_columns: Columns which values will be based on the right set.
        :type right_columns: list
        :param output_set: Name of the output annotations set.
        :type output_set: str
        :return: [description]
        :rtype: [type]
        """
        assert left_set != right_set, "sets must differ"
        assert not (set(left_columns) & set (right_columns)), "left_columns and right_columns must be disjoint"

        union = set(left_columns) | set (right_columns)
        all_columns = {c.name for c in self.SEGMENTS_COLUMNS} - {'raw_filename', 'segment_onset', 'segment_offset'}
        required_columns = {c.name for c in self.SEGMENTS_COLUMNS if c.required} - {'raw_filename', 'segment_onset', 'segment_offset'}
        assert union.issubset(all_columns), "left_columns and right_columns have unexpected values"
        assert required_columns.issubset(union), "left_columns and right_columns have missing values"

        left_annotations = self.annotations[self.annotations['set'] == left_set]
        right_annotations = self.annotations[self.annotations['set'] == right_set]

        left_annotations = left_annotations[left_annotations['error'].isnull()]
        right_annotations = right_annotations[right_annotations['error'].isnull()]

        left_annotations, right_annotations = self.intersection(left_annotations, right_annotations)
        left_annotations = left_annotations.reset_index(drop = True).rename_axis('interval').reset_index()
        right_annotations = right_annotations.reset_index(drop = True).rename_axis('interval').reset_index()

        input_annotations = [
            {
                'left_annotations': left_annotations[left_annotations['recording_filename'] == recording],
                'right_annotations': right_annotations[right_annotations['recording_filename'] == recording]
            }
            for recording in left_annotations['recording_filename'].unique()
        ]
            
        pool = mp.Pool(processes = threads if threads > 0 else mp.cpu_count())
        annotations = pool.map(partial(self.merge_annotations, left_columns, right_columns, columns, output_set), input_annotations)
        annotations = pd.concat(annotations)
        annotations.drop(columns = list(set(annotations.columns)-{c.name for c in self.INDEX_COLUMNS}), inplace = True)
        annotations.fillna({'raw_filename': 'NA'}, inplace = True)
        
        self.read()
        self.annotations = pd.concat([self.annotations, annotations], sort = False)
        self.annotations.to_csv(os.path.join(self.project.path, 'metadata/annotations.csv'), index = False)

    def get_segments(self, annotations: pd.DataFrame) -> pd.DataFrame:
        """get all segments associated to the annotations referenced in ``annotations``.

        :param annotations: dataframe of annotations, according to :ref:`format-annotations`
        :type annotations: pd.DataFramef
        :return: dataframe of all the segments merged (as specified in :ref:`format-annotations-segments`), merged with ``annotations``. 
        :rtype: pd.DataFrame
        """
        annotations = annotations.dropna(subset = ['annotation_filename'])
        annotations.drop(columns = ['raw_filename'], inplace = True)

        segments = pd.concat([
            pd.read_csv(os.path.join(self.project.path, 'annotations', a['set'], 'converted', a['annotation_filename'])).assign(set = a['set'], annotation_filename = a['annotation_filename'])
            for a in annotations.to_dict(orient = 'records')
        ])

        return segments.merge(annotations, how = 'left', left_on = ['set', 'annotation_filename'], right_on = ['set', 'annotation_filename'])

    def intersection(self, left: pd.DataFrame, right: pd.DataFrame) -> tuple:
        """Compute the intersection of all ``left`` and ``right`` annotations,
        based on their ``recording_filename``, ``time_seek``, ``range_onset`` and ``range_offset``
        attributes. (Only these columns are required, but more can be passed and they
        will be preserved).

        :param left: dataframe of annotations, according to :ref:`format-annotations`
        :type left: pd.DataFrame
        :param right: dataframe of annotations, according to :ref:`format-annotations`
        :type right: pd.DataFrame
        :return: dataframe of annotations, according to :ref:`format-annotations`
        :rtype: tuple
        """
        recordings = set(left['recording_filename'].unique()) & set(right['recording_filename'].unique())
        recordings = list(recordings)

        a_stack = []
        b_stack = []

        for recording in recordings:
            a = left[left['recording_filename'] == recording].copy()
            b = right[right['recording_filename'] == recording].copy()

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

    def clip_segments(self, segments: pd.DataFrame, start: int, stop: int) -> pd.DataFrame:
        """Clip all segments onsets and offsets within ``start`` and ``stop``.
        Segments outside of the range [``start``,``stop``] will be removed.

        :param segments: Dataframe of the segments to clip
        :type segments: pd.DataFrame
        :param start: range start (in milliseconds)
        :type start: int
        :param stop: range end (in milliseconds)
        :type stop: int
        :return: Dataframe of the clipped segments
        :rtype: pd.DataFrame
        """
        start = int(start)
        stop = int(stop)

        segments['segment_onset'].clip(lower = start, upper = stop, inplace = True)
        segments['segment_offset'].clip(lower = start, upper = stop, inplace = True)

        segments = segments[(segments['segment_offset'] - segments['segment_onset']) > 0]

        return segments

    def get_vc_stats(self, segments: pd.DataFrame, turntakingthresh: int = 1000):
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
        ).astype(int)
