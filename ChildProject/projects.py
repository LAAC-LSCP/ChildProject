import datetime
from functools import partial, reduce
import glob
import multiprocessing as mp
import numpy as np
import operator
import os
import pandas as pd
import re
import shutil
import subprocess

from .tables import IndexTable, IndexColumn

class RecordingProfile:
    def __init__(self, name, format = 'wav', codec = 'pcm_s16le', sampling = 16000,
                 split = None, extra_flags = None):

        self.name = str(name)
        self.format = format
        self.codec = str(codec)
        self.sampling = int(sampling)
        self.extra_flags = extra_flags

        if split is not None:
            try:
                split_time = datetime.datetime.strptime(split, '%H:%M:%S')
            except:
                raise ValueError('split should be specified as HH:MM:SS')

        self.split = split
        self.recordings = []

    def to_csv(self, destination):
        pd.DataFrame([
            {'key': 'name', 'value': self.name},
            {'key': 'format', 'value': self.format},
            {'key': 'codec', 'value': self.codec},
            {'key': 'sampling', 'value': self.sampling},
            {'key': 'split', 'value': self.split},
            {'key': 'extra_flags', 'value': self.extra_flags}
        ]).to_csv(destination, index = False)

def convert_recording(path, profile, skip_existing, row):
    if row['filename'] == 'NA':
            return []

    original_file = os.path.join(
        path,
        'recordings',
        row['filename']
    )

    destination_file = os.path.join(
        path,
        'converted_recordings',
        profile.name,
        os.path.splitext(row['filename'])[0] + '.%03d.' + profile.format if profile.split
        else os.path.splitext(row['filename'])[0] + '.' + profile.format
    )

    os.makedirs(
        name = os.path.dirname(destination_file),
        exist_ok = True
    )

    skip = skip_existing and os.path.exists(destination_file)
    success = skip

    if not skip:
        split_args = []
        if profile.split:
            split_args.append('-segment_time')
            split_args.append(profile.split)
            split_args.append('-f')
            split_args.append('segment')

        proc = subprocess.Popen(
            [
                'ffmpeg', '-y',
                '-loglevel', 'error',
                '-i', original_file,
                '-ac', '1',
                '-c:a', profile.codec,
                '-ar', str(profile.sampling)
            ]
            + split_args
            + [
                destination_file
            ],
            stdout = subprocess.DEVNULL,
            stderr = subprocess.PIPE
        )
        (stdout, stderr) = proc.communicate()

        success = proc.returncode == 0

    if not success:
        return [{
            'original_filename': row['filename'],
            'converted_filename': "",
            'success': False,
            'error': stderr
        }]
    else:
        if profile.split:
            converted_files = [
                os.path.basename(cf)
                for cf in glob.glob(os.path.join(path, 'converted_recordings', profile.name, os.path.splitext(row['filename'])[0] + '.*.' + profile.format))
            ]
        else:
            converted_files = [os.path.splitext(row['filename'])[0] + '.' + profile.format]

    return [{
        'original_filename': row['filename'],
        'converted_filename': cf,
        'success': True
    } for cf in converted_files]

class ChildProject:
    REQUIRED_DIRECTORIES = [
        'recordings',
        'extra'
    ]

    CHILDREN_COLUMNS = [
        IndexColumn(name = 'experiment', description = 'one word to capture the unique ID of the data collection effort; for instance Tsimane_2018, solis-intervention-pre', required = True),
        IndexColumn(name = 'child_id', description = 'unique child ID -- unique within the experiment (Id could be repeated across experiments to refer to different children)', unique = True, required = True),
        IndexColumn(name = 'child_dob', description = "child's date of birth", required = True, datetime = '%Y-%m-%d'),
        IndexColumn(name = 'location_id', description = 'Unique location ID -- only specify here if children never change locations in this culture; otherwise, specify in the recordings metadata'),
        IndexColumn(name = 'child_sex', description = 'f= female, m=male', regex = '(m|f|M|F)'),
        IndexColumn(name = 'language', description = 'language the child is exposed to if child is monolingual; small caps, indicate dialect by name or location if available; eg "france french"; "paris french"'),
        IndexColumn(name = 'languages', description = 'list languages child is exposed to separating them with ; and indicating the percentage if one is available; eg: "french 35%; english 65%"'),
        IndexColumn(name = 'mat_ed', description = 'maternal years of education'),
        IndexColumn(name = 'fat_ed', description = 'paternal years of education'),
        IndexColumn(name = 'car_ed', description = 'years of education of main caregiver (if not mother or father)'),
        IndexColumn(name = 'monoling', description = 'whether the child is monolingual (Y) or not (N)', regex = '(Y|N)'),
        IndexColumn(name = 'monoling_criterion', description = 'how monoling was decided; eg "we asked families which languages they spoke in the home"'),
        IndexColumn(name = 'normative', description = 'whether the child is normative (Y) or not (N)', regex = '(Y|N)'),
        IndexColumn(name = 'normative_criterion', description = 'how normative was decided; eg "unless the caregivers volunteered information whereby the child had a problem, we consider them normative by default"'),
        IndexColumn(name = 'mother_id', description = 'unique ID of the mother'),
        IndexColumn(name = 'father_id', description = 'unique ID of the father'),
        IndexColumn(name = 'order_of_birth', description = 'child order of birth', regex = '([0-9]+|NA)', required = False),
        IndexColumn(name = 'n_of_siblings', description = 'amount of siblings', regex = '([0-9]+|NA)', required = False),
        IndexColumn(name = 'household_size', description = 'household size', regex = '([0-9]+|NA)', required = False),
        IndexColumn(name = 'dob_criterion', description = "determines whether the date of birth is known exactly or extrapolated e.g. from the age. Dates of birth are assumed to be known exactly if this column is NA or unspecified.", regex = "(extrapolated|exact|NA)", required = False)
    ]

    RECORDINGS_COLUMNS = [
        IndexColumn(name = 'experiment', description = 'one word to capture the unique ID of the data collection effort; for instance Tsimane_2018, solis-intervention-pre', required = True),
        IndexColumn(name = 'child_id', description = 'unique child ID -- unique within the experiment (Id could be repeated across experiments to refer to different children)', required = True),
        IndexColumn(name = 'date_iso', description = 'date in which recording was started in ISO (eg 2020-09-17)', required = True, datetime = '%Y-%m-%d'),
        IndexColumn(name = 'start_time', description = 'local time in which recording was started in format 24-hour (H)H:MM; if minutes are unknown, use 00. Set as ‘NA’ if unknown.', required = True, datetime = '%H:%M'),
        IndexColumn(name = 'recording_device_type', description = 'lena, usb, olympus, babylogger (lowercase)', required = True, regex = '({})'.format('|'.join(['lena', 'usb', 'olympus', 'babylogger']))),
        IndexColumn(name = 'filename', description = 'the path to the file from the root of “recordings”), set to ‘NA’ if no valid recording available. It is unique (two recordings cannot point towards the same file).', required = True, filename = True, unique = True),
        IndexColumn(name = 'recording_device_id', description = 'unique ID of the recording device'),
        IndexColumn(name = 'experimenter', description = 'who collected the data (could be anonymized ID)'),
        IndexColumn(name = 'location_id', description = 'unique location ID -- can be specified at the level of the child (if children do not change locations)'),
        IndexColumn(name = 'its_filename', description = 'its_filename', filename = True),
        IndexColumn(name = 'upl_filename', description = 'upl_filename', filename = True),
        IndexColumn(name = 'lena_id', description = ''),
        IndexColumn(name = 'notes', description = 'free-style notes about individual recordings (avoid tabs and newlines)'),
        IndexColumn(name = 'daytime', description = 'yes (Y) means recording launched such that most or all of the audiorecording happens during daytime; no (N) means at least 30% of the recording may happen at night', regex = '(Y|N)')
    ]

    PROJECT_FOLDERS = [
        'raw_annotations',
        'annotations',
        'converted_recordings',
        'doc',
        'scripts'
    ]

    def __init__(self, path):
        self.path = path
        self.errors = []
        self.warnings = []
        self.children = None
        self.recordings = None
    
    def read(self):
        self.ct = IndexTable('children', os.path.join(self.path, 'children'), self.CHILDREN_COLUMNS)
        self.rt = IndexTable('recordings', os.path.join(self.path, 'recordings/recordings'), self.RECORDINGS_COLUMNS)

        self.children = self.ct.read(lookup_extensions = ['.csv', '.xls', '.xlsx'])
        self.recordings = self.rt.read(lookup_extensions = ['.csv', '.xls', '.xlsx'])

    def validate_input_data(self):
        self.errors = []
        self.warnings = []

        path = self.path

        directories = [d for d in os.listdir(path) if os.path.isdir(path)]

        for rd in self.REQUIRED_DIRECTORIES:
            if rd not in directories:
                self.errors.append("missing directory {}.".format(rd))

        # check tables
        self.read()
        
        errors, warnings = self.ct.validate()
        self.errors += errors
        self.warnings += warnings

        errors, warnings = self.rt.validate()
        self.errors += errors
        self.warnings += warnings

        for index, row in self.recordings.iterrows():
            # make sure that recordings exist
            for column_name in self.recordings.columns:
                column_attr = next((c for c in self.RECORDINGS_COLUMNS if c.name == column_name), None)

                if column_attr is None:
                    continue

                if column_attr.filename and row[column_name] != 'NA' and not os.path.exists(os.path.join(path, 'recordings', str(row[column_name]))):
                    self.errors.append("cannot find recording '{}'".format(str(row[column_name])))

            # child id refers to an existing child in the children table
            if row['child_id'] not in self.children['child_id'].tolist():
                self.errors.append("child_id '{}' in recordings on line {} cannot be found in the children table.".format(row['child_id'], index))

        # detect un-indexed recordings and throw warnings
        files = [
            self.recordings[c.name].tolist()
            for c in self.RECORDINGS_COLUMNS
            if c.filename and c.name in self.recordings.columns
        ]

        indexed_files = [
            os.path.abspath(os.path.join(path, 'recordings', str(f)))
            for f in pd.core.common.flatten(files)
        ]

        recordings_files = glob.glob(os.path.join(path, 'recordings', '**/*.*'), recursive = True)

        for rf in recordings_files:
            if len(os.path.splitext(rf)) > 1 and os.path.splitext(rf)[1] in ['.csv', '.xls', '.xlsx']:
                continue

            ap = os.path.abspath(rf)
            if ap not in indexed_files:
                self.warnings.append("file '{}' not indexed.".format(rf))

        return self.errors, self.warnings

    def import_data(self, destination, follow_symlinks = True):
        errors, warnings = self.validate_input_data()

        if len(errors) > 0:
            raise Exception('cannot import data: validation failed')

        # perform copy
        shutil.copytree(src = self.path, dst = destination, symlinks = follow_symlinks)

        # create folders
        for folder in self.PROJECT_FOLDERS:
            os.makedirs(
                name = os.path.join(destination, folder),
                exist_ok = True
            )

    def convert_recordings(self, profile, skip_existing = False, threads = 0):
        if not isinstance(profile, RecordingProfile):
            raise ValueError('profile should be a RecordingProfile instance')

        errors, warnings = self.validate_input_data()
        if len(errors) > 0:
            raise Exception('cannot convert: validation failed')

        os.makedirs(
            name = os.path.join(self.path, 'converted_recordings', profile.name),
            exist_ok = True
        )

        conversion_table = []

        pool = mp.Pool(processes = threads if threads > 0 else mp.cpu_count())
        conversion_table = pool.map(
            partial(convert_recording, self.path, profile, skip_existing),
            self.recordings.to_dict('records')
        )

        conversion_table = reduce(operator.concat, conversion_table)
        profile.recordings = pd.DataFrame(conversion_table)

        profile.recordings.to_csv(os.path.join(self.path, 'converted_recordings', profile.name, 'recordings.csv'), index = False)
        profile.to_csv(os.path.join(self.path, 'converted_recordings', profile.name, 'profile.csv'))

        return profile