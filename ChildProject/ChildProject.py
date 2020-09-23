import pandas as pd
import numpy as np
import os
import datetime
import glob
import shutil
import subprocess

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

class ChildProject:
    REQUIRED_DIRECTORIES = [
        'recordings',
        'extra'
    ]

    CHILDREN_REQUIRED_COLUMNS = [
        'experiment',
        'child_id',
        'child_dob',
        'lineno' # generated
    ]

    CHILDREN_OPTIONAL_COLUMNS = [
        'location_id',
        'culture',
        'child_sex',
        'language',
        'languages',
        'mat_ed',
        'fat_ed',
        'car_ed',
        'monoling',
        'monoling_criterion',
        'normative',
        'normative_criterion',
        'mother_id',
        'father_id',
        'daytime'
    ]

    RECORDINGS_REQUIRED_COLUMNS = [
        'experiment',
        'child_id',
        'date_iso',
        'start_time',
        'recording_device_type',
        'filename',
        'lineno' # generated
    ]

    RECORDINGS_OPTIONAL_COLUMNS = [
        'recording_device_id',
        'experimenter',
        'location_id',
        'its_filename',
        'upl_filename',
        'lena_id',
        'age',
        'notes'
    ]

    DATE_FORMAT = '%Y-%m-%d'
    TIME_FORMAT = '%H:%M'

    RECORDING_DEVICE_TYPES = [
        'usb',
        'olympus',
        'lena',
        'babylogger'
    ]

    PROJECT_FOLDERS = [
        'doc',
        'scripts',
        'converted_recordings'
    ]

    def __init__(self, path):
        self.path = path
        self.errors = []
        self.warnings = []
        self.children = None
        self.recordings = None

    def register_error(self, message, fatal = False):
        if fatal:
            raise Exception(message)

        self.errors.append(message)

    def register_warning(self, message):
        self.warnings.append(message)

    def read_table(self, path):
        extensions = ['.csv', '.xls', '.xlsx']
        
        for extension in extensions:
            filename = path + extension
            
            if not os.path.exists(filename):
                continue

            pd_flags = {
                'keep_default_na': False,
                'na_values': ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN',
                              '#N/A N/A', '#N/A', 'N/A', 'n/a', '', '#NA',
                              'NULL', 'null', 'NaN', '-NaN', 'nan',
                              '-nan', '']
            }

            try:
                if extension == '.csv':
                    df = pd.read_csv(filename, **pd_flags)
                elif extension == '.xls' or extension == '.xlsx':
                    df = pd.read_excel(filename, **pd_flags)
                else:
                    raise Exception('table format not supported ({})'.format(extension))    
            except Exception as e:
                self.register_error(str(e), True)
                return None

            df['lineno'] = df.index + 2
            return df

        self.register_error("could not find table '{}'".format(path), True)
        return None
    
    def read(self):
        self.children = self.read_table(os.path.join(self.path, 'children'))
        self.recordings = self.read_table(os.path.join(self.path, 'recordings/recordings'))

    def validate_input_data(self):
        self.errors = []
        self.warnings = []

        path = self.path

        directories = [d for d in os.listdir(path) if os.path.isdir(path)]

        for rd in self.REQUIRED_DIRECTORIES:
            if rd not in directories:
                self.register_error("missing directory {}.".format(rd), True)

        # check tables
        self.read()

        for rc in self.CHILDREN_REQUIRED_COLUMNS:
            if rc not in self.children.columns:
                self.register_error("children table is missing column '{}'".format(rc), True)

            null = self.children[self.children[rc].isnull()]['lineno'].tolist()
            if len(null) > 0:
                self.register_error(
                    """children table has undefined values
                    for column '{}' in lines: {}""".format(rc, ','.join(null))
                )
        
        unknown_columns = [
            c for c in self.children.columns
            if c not in self.CHILDREN_REQUIRED_COLUMNS and c not in self.CHILDREN_OPTIONAL_COLUMNS
        ]

        if len(unknown_columns) > 0:
            self.register_warning("unknown column{} '{}' in children, exepected columns are: {}".format(
                's' if len(unknown_columns) > 1 else '',
                ','.join(unknown_columns),
                ','.join(self.CHILDREN_REQUIRED_COLUMNS+self.CHILDREN_OPTIONAL_COLUMNS)
            ))


        for rc in self.RECORDINGS_REQUIRED_COLUMNS:
            if rc not in self.recordings.columns:
                self.register_error("recordings table is missing column '{}'".format(rc), True)

            null = self.recordings[self.recordings[rc].isnull()]['lineno'].tolist()
            if len(null) > 0:
                self.register_error(
                    """recordings table has undefined values
                    for column '{}' in lines: {}""".format(rc, ','.join(map(str, null)))
                )

        unknown_columns = [
            c for c in self.recordings.columns
            if c not in self.RECORDINGS_REQUIRED_COLUMNS and c not in self.RECORDINGS_OPTIONAL_COLUMNS
        ]

        if len(unknown_columns) > 0:
            self.register_warning("unknown column{} '{}' in recordings, exepected columns are: {}".format(
                's' if len(unknown_columns) > 1 else '',
                ','.join(unknown_columns),
                ','.join(self.RECORDINGS_REQUIRED_COLUMNS+self.RECORDINGS_OPTIONAL_COLUMNS)
            ))

        for index, row in self.recordings.iterrows():
            # make sure that recordings exist
            if row['filename'] != 'NA' and not os.path.exists(os.path.join(path, 'recordings', str(row['filename']))):
                self.register_error("cannot find recording '{}'".format(str(row['filename'])))

            # date is valid
            try:
                date = datetime.datetime.strptime(row['date_iso'], self.DATE_FORMAT)
            except:
                self.register_error("'{}' is not a proper date (expected YYYY-MM-DD) on line {}".format(row['date_iso'], row['lineno']))

            # start_time is valid
            if row['start_time'] != 'NA':
                try:
                    start = datetime.datetime.strptime("1970-01-01 {}".format(str(row['start_time'])[:5]), "%Y-%m-%d %H:%M")
                except:
                    self.register_error("'{}' is not a proper time (expected HH:MM) on line {}".format(str(row['start_time'])[:5], row['lineno']))

            # child id refers to an existing child in the children table
            if row['child_id'] not in self.children['child_id'].tolist():
                self.register_error("child_id '{}' in recordings on line {} cannot be found in the children table.".format(row['child_id'], row['lineno']))

            # recording_device_type exists
            if row['recording_device_type'] not in self.RECORDING_DEVICE_TYPES:
                self.register_warning("invalid device type '{}' in recordings on line {}".format(row['recording_device_type'], row['lineno']))

        # look for duplicates
        grouped = self.recordings[self.recordings['filename'] != 'NA']\
            .groupby('filename')['lineno']\
            .agg([
                ('count', len),
                ('lines', lambda lines: ",".join([str(line) for line in sorted(lines)])),
                ('first', np.min)
            ])\
            .sort_values('first')

        duplicates = grouped[grouped['count'] > 1]
        for filename, row in duplicates.iterrows():
            self.register_error("filename '{}' appears {} times in lines [{}], should appear once".format(
                filename,
                row['count'],
                row['lines']
            ))


        # detect un-indexed recordings and throw warnings
        self.recordings['abspath'] = self.recordings['filename'].apply(lambda s:
            os.path.abspath(os.path.join(path, 'recordings', str(s)))   
        )

        recordings_files = glob.glob(os.path.join(path, 'recordings', '**.*'), recursive = True)

        for rf in recordings_files:
            if len(os.path.splitext(rf)) > 1 and os.path.splitext(rf)[1] in ['.csv', '.xls', '.xlsx']:
                continue

            ap = os.path.abspath(rf)
            if ap not in self.recordings['abspath'].tolist():
                self.register_warning("file '{}' not indexed.".format(rf))

        return {
            'errors': self.errors,
            'warnings': self.warnings
        }

    def import_data(self, destination, follow_symlinks = True):
        validation = self.validate_input_data()

        if len(validation['errors']) > 0:
            raise Exception('cannot import data: validation failed')

        # perform copy
        shutil.copytree(src = self.path, dst = destination, symlinks = follow_symlinks)

        # create folders
        for folder in self.PROJECT_FOLDERS:
            os.makedirs(
                name = os.path.join(destination, folder),
                exist_ok = True
            )

    def convert_recordings(self, profile):
        if not isinstance(profile, RecordingProfile):
            raise ValueError('profile should be a RecordingProfile instance')

        validation = self.validate_input_data()
        if len(validation['errors']) > 0:
            raise Exception('cannot convert: validation failed')

        os.makedirs(
            name = os.path.join(self.path, 'converted_recordings', profile.name),
            exist_ok = True
        )

        conversion_table = []

        for index, row in self.recordings.iterrows():
            if row['filename'] == 'NA':
                continue

            original_file = os.path.join(
                self.path,
                'recordings',
                row['filename']
            )

            destination_file = os.path.join(
                self.path,
                'converted_recordings',
                profile.name,
                os.path.splitext(row['filename'])[0] + '.%03d.' + profile.format if profile.split
                else os.path.splitext(row['filename'])[0] + '.' + profile.format
            )

            split_args = []
            if profile.split:
                split_args.append('-segment_time')
                split_args.append(profile.split)
                split_args.append('-f')
                split_args.append('segment')

            proc = subprocess.Popen(
                [
                    'ffmpeg',
                    '-y',
                    '-i',
                    original_file,
                    '-ac',
                    '1',
                    '-c:a',
                    profile.codec,
                    '-ar',
                    str(profile.sampling)
                ]
                + split_args
                + [
                    destination_file
                ],
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE
            )
            proc.wait()
            (stdout, stderr) = proc.communicate()

            if proc.returncode != 0:
                self.register_error("failure while processing recording '{}': {}".format(row['filename'], str(stderr)))
                conversion_table.append({
                    'original_filename': row['filename'],
                    'converted_filename': "",
                    'success': False
                })
            else:
                if profile.split:
                    converted_files = [
                        os.path.basename(cf)
                        for cf in glob.glob(os.path.join(self.path, 'converted_recordings', profile.name, os.path.splitext(row['filename'])[0] + '.*.' + profile.format))
                    ]
                else:
                    converted_files = [os.path.splitext(row['filename'])[0] + '.' + profile.format]

                conversion_table += [{
                    'original_filename': row['filename'],
                    'converted_filename': cf,
                    'success': True
                } for cf in converted_files]

        profile.recordings = pd.DataFrame(conversion_table)

        profile.recordings.to_csv(
            os.path.join(
                self.path,
                'converted_recordings',
                profile.name,
                'recordings.csv'
            ),
            index = False
        )

        profile.to_csv(os.path.join(
            self.path,
            'converted_recordings',
            profile.name,
            'profile.csv'
        ))

        return profile