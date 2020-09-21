import pandas as pd
import numpy as np
import os
import datetime
import glob

class ChildProject:
    REQUIRED_DIRECTORIES = [
        'recordings',
        'extra'
    ]

    CHILDREN_REQUIRED_COLUMNS = [
        'experiment',
        'child_id'
    ]

    RECORDINGS_REQUIRED_COLUMNS = [
        'experiment',
        'child_id',
        'date_iso',
        'start_time',
        'recording_device_type',
        'filename'
    ]

    DATE_FORMAT = '%Y-%m-%d'
    TIME_FORMAT = '%H:%M'

    RECORDING_DEVICE_TYPES = [
        'usb',
        'olympus',
        'lena',
        'babylogger'
    ]

    def __init__(self):
        self.raw_data_path = ''
        self.errors = []
        self.warnings = []

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
    

    def validate_input_data(self):
        self.errors = []
        self.warnings = []

        path = self.raw_data_path

        directories = [d for d in os.listdir(path) if os.path.isdir(path)]

        for rd in self.REQUIRED_DIRECTORIES:
            if rd not in directories:
                self.register_error("missing directory {}.".format(rd), True)

        # check tables
        self.children = self.read_table(os.path.join(path, 'children'))

        for rc in self.CHILDREN_REQUIRED_COLUMNS:
            if rc not in self.children.columns:
                self.register_error("children table is missing column '{}'".format(rc), True)

            null = self.children[self.children[rc].isnull()]['lineno'].tolist()
            if len(null) > 0:
                self.register_error(
                    """children table has undefined values
                    for column '{}' in lines: {}""".format(rc, ','.join(null))
                )         

        self.recordings = self.read_table(os.path.join(path, 'recordings/recordings'))
        for rc in self.RECORDINGS_REQUIRED_COLUMNS:
            if rc not in self.recordings.columns:
                self.register_error("recordings table is missing column '{}'".format(rc))

            null = self.recordings[self.recordings[rc].isnull()]['lineno'].tolist()
            if len(null) > 0:
                self.register_error(
                    """recordings table has undefined values
                    for column '{}' in lines: {}""".format(rc, ','.join(map(str, null)))
                )

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

