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

from ChildProject.projects import ChildProject
from ChildProject.pipelines.pipeline import Pipeline

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

class ConversionPipeline(Pipeline):

    def run(self, path, name, format, codec, sampling, split = None, skip_existing = False, threads = 0, **kwargs):
        profile = RecordingProfile(
            name,
            format = format,
            codec = codec,
            sampling = sampling,
            split = split,
            extra_flags = None
        )

        project = ChildProject(path)

        errors, warnings = project.validate_input_data()
        if len(errors) > 0:
            raise Exception('cannot convert: validation failed')

        os.makedirs(
            name = os.path.join(project.path, 'converted_recordings', profile.name),
            exist_ok = True
        )

        conversion_table = []

        pool = mp.Pool(processes = threads if threads > 0 else mp.cpu_count())
        conversion_table = pool.map(
            partial(convert_recording, project.path, profile, skip_existing),
            project.recordings.to_dict('records')
        )

        conversion_table = reduce(operator.concat, conversion_table)
        profile.recordings = pd.DataFrame(conversion_table)

        profile.recordings.to_csv(os.path.join(project.path, 'converted_recordings', profile.name, 'recordings.csv'), index = False)
        profile.to_csv(os.path.join(project.path, 'converted_recordings', profile.name, 'profile.csv'))

        return profile

    @staticmethod
    def setup_parser(parser):
        default_profile = RecordingProfile(name = None)
        parser.add_argument("path", help = "project path")
        parser.add_argument("--name", help = "profile name", required = True)
        parser.add_argument("--format", help = "audio format (e.g. {})".format(default_profile.format), required = True)
        parser.add_argument("--codec", help = "audio codec (e.g. {})".format(default_profile.codec), required = True)
        parser.add_argument("--sampling", help = "sampling frequency (e.g. {})".format(default_profile.sampling), required = True)
        parser.add_argument("--split", help = "split duration (e.g. 15:00:00)", required = False, default = None)
        parser.add_argument('--skip-existing', dest='skip_existing', required = False, default = False, action='store_true')
        parser.add_argument('--threads', help = "amount of threads running conversions in parallel (0 = uses all available cores)", required = False, default = 0, type = int)
