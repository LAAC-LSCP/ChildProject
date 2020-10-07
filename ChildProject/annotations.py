import pandas as pd
import shutil
import os

from .projects import ChildProject
from .tables import IndexTable, IndexColumn

class AnnotationProcess:
    INPUT_COLUMNS = [
        IndexColumn(name = 'recording_filename', description = 'recording filename as in the recordings index', required = True),
        IndexColumn(name = 'segment_offset', description = 'extract start time in seconds, e.g: 3600, or 3600.500', regex = r"[0-9]{1,}(\.[0-9]{3})?", required = False),
        IndexColumn(name = 'segment_length', description = 'extract length in seconds, e.g: 3600, or 3600.500', regex = r"[0-9]{1,}(\.[0-9]{3})?", required = False)    ]

    INDEX_COLUMNS = INPUT_COLUMNS + [
        IndexColumn(name = 'input_filename', description = 'file to be processed', required = True),
        IndexColumn(name = 'completed', description = 'processing completed for this file', required = True, regex = r'(True|False)'),
        IndexColumn(name = 'result', description = 'processing output', required = False),
        IndexColumn(name = 'profile', description = 'audio profile', required = False)

    ]

    def __init__(self, name, project):
        self.name = name
        self.project = project

        if not isinstance(project, ChildProject):
            raise ValueError('project should derive from ChildProject')

        project.read()

        os.makedirs(os.path.join('annotations', self.name, 'input'), exist_ok = True)
        os.makedirs(os.path.join('annotations', self.name, 'output'), exist_ok = True)

        index_path = os.path.join(self.project.path, 'annotations', self.name, 'index.csv')
        if not os.path.exists(input_index_path):
            open(input_index_path, 'w+').write(','.join([c.name for c in self.INPUT_COLUMNS]))

    def read_index(self):
        index = IndexTable(
            'index',
            path = os.path.join(self.project.path, 'annotations', self.name, 'index.csv'),
            columns = self.INDEX_COLUMNS
        )

        df = index.read()
        return df

    def extract_audio(self, filename, offset, length, destination):
        return None
        
    def pre_process(self, input_df, profile = None):
        it = IndexTable('input', columns = self.INPUT_COLUMNS)
        it.df = input_df
        errors, warnings = it.validate()

        if profile:
            profile_df = pd.read_csv(os.path.join(self.project.path, 'converted_recordings', profile, 'recordings.csv'))
            input_df = input_df.merge(profile_df, left_on = 'recording_filename', right_on = 'original_filename')

        for index, row in input_df.iterrows():
            if profile:
                if not row['converted_filename'] or not row['success']:
                    errors.append("could not find any converted version of '{}' for profile '{}'".format(row['recording_filename'], profile))
            else:
                if row['recording_filename'] not in self.project.recordings['filename'].tolist():
                    errors.append("'{}' is not a valid recording filename".format(row['recording_filename']))

        if len(errors) > 0:
            return errors, warnings

        preprocessed_inputs = []
        for index, row in input_df.iterrows():
            if profile:
                origin = os.path.join(self.project.path, 'converted_recordings', profile, row['converted_filename'])
            else:
                origin = os.path.join(self.project.path, 'recordings', row['recording_filename'])
            
            if row['segment_offset'] and row['segment_length']:
                base, extension = os.path.splitext(row['recording_filename'])

                filename = "{}_{}_{}{}".format(base, row['segment_offset'], row['segment_length'], extension)

                self.extract_audio(
                    origin,
                    row['segment_offset'],
                    row['segment_length'],
                    os.path.join(self.project.path, 'annotations', self.name, 'input', filename)
                )
            else:
                filename = row['recording_filename']
                shutil.copyfile(
                    origin,
                    os.path.join(self.project.path, 'annotations', self.name, 'input', filename)
                )
            
            preprocessed_inputs.append(dict(row).update({
                'input_filename': filename,
                'completed': False,
                'profile': profile
            }))

        df = self.read_index()

        merge_cols = ['recording_filename', 'segment_offset', 'segment_length']
        df = df.merge(pd.DataFrame(preprocessed_inputs), how = 'outer', left_on = merge_cols, right_on = merge_cols)
        df.to_csv(os.path.join(self.project.path, 'annotations', self.name, 'index.csv'), index = False)

        return errors, warnings

    def get_queue(self):
        index = IndexTable(
            'index',
            path = os.path.join(self.project.path, 'annotations', self.name, 'index.csv'),
            columns = self.INDEX_COLUMNS
        )

        queue = index.read()
        queue = queue[queue['completed'] == False]

        return queue

    def process_segment(self, segment, options):
        return 0

    def process(self, options):
        queue = self.get_queue()

        queue['result'] = queue.apply(lambda row: self.process_segment(row, options))
        queue['completed'] = queue['result'] == 0

        merge_cols = ['recording_filename', 'segment_offset', 'segment_length']
        df = self.read_index()
        df = df.merge(queue, how = 'outer', left_on = merge_cols, right_on = merge_cols)
        df.to_csv(os.path.join(self.project.path, 'annotations', self.name, 'index.csv'), index = False)
        return None

    def post_process(self):
        return None






