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
    def __init__(self, name: str, format: str = 'wav', codec: str = 'pcm_s16le', sampling: int = 16000,
                 split: str = None, extra_flags: str = None):

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

    def to_csv(self, destination: str):
        pd.DataFrame([
            {'key': 'name', 'value': self.name},
            {'key': 'format', 'value': self.format},
            {'key': 'codec', 'value': self.codec},
            {'key': 'sampling', 'value': self.sampling},
            {'key': 'split', 'value': self.split},
            {'key': 'extra_flags', 'value': self.extra_flags}
        ]).to_csv(destination, index = False)

def convert_recording(path: str, profile: str, skip_existing: bool, row: dict) -> list:
    if row['recording_filename'] == 'NA':
            return []

    original_file = os.path.join(
        path,
        ChildProject.RAW_RECORDINGS,
        row['recording_filename']
    )

    destination_file = os.path.join(
        path,
        ChildProject.CONVERTED_RECORDINGS,
        profile.name,
        os.path.splitext(row['recording_filename'])[0] + '.%03d.' + profile.format if profile.split
        else os.path.splitext(row['recording_filename'])[0] + '.' + profile.format
    )

    os.makedirs(
        name = os.path.dirname(destination_file),
        exist_ok = True
    )

    skip = skip_existing and os.path.exists(destination_file)
    success = skip

    if not skip:
        args = [
            'ffmpeg', '-y',
            '-loglevel', 'error',
            '-i', original_file,
            '-c:a', profile.codec,
            '-ar', str(profile.sampling)
        ]

        if profile.split:
            args.extend([
                '-segment_time', profile.split, '-f', 'segment'
            ])

        args.append(destination_file)

        proc = subprocess.Popen(args, stdout = subprocess.DEVNULL, stderr = subprocess.PIPE)
        (stdout, stderr) = proc.communicate()
        success = proc.returncode == 0

    if not success:
        return [{
            'original_filename': row['recording_filename'],
            'converted_filename': "",
            'success': False,
            'error': stderr
        }]
    else:
        if profile.split:
            converted_files = [
                os.path.basename(cf)
                for cf in glob.glob(os.path.join(path, ChildProject.CONVERTED_RECORDINGS, profile.name, os.path.splitext(row['recording_filename'])[0] + '.*.' + profile.format))
            ]
        else:
            converted_files = [os.path.splitext(row['recording_filename'])[0] + '.' + profile.format]

    return [{
        'original_filename': row['recording_filename'],
        'converted_filename': cf,
        'success': True
    } for cf in converted_files]

class ConversionPipeline(Pipeline):

    def run(self, path: str, name: str, format: str, codec: str, sampling: int, split: str = None, skip_existing: bool = False, threads: int = 0, **kwargs) -> RecordingProfile:
        profile = RecordingProfile(
            name,
            format = format,
            codec = codec,
            sampling = sampling,
            split = split,
            extra_flags = None
        )

        project = ChildProject(path)

        errors, warnings = project.validate()
        if len(errors) > 0:
            raise Exception('cannot convert: validation failed')

        os.makedirs(
            name = os.path.join(project.path, ChildProject.CONVERTED_RECORDINGS, profile.name),
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

        profile.recordings.to_csv(os.path.join(project.path, ChildProject.CONVERTED_RECORDINGS, profile.name, 'recordings.csv'), index = False)
        profile.to_csv(os.path.join(project.path, ChildProject.CONVERTED_RECORDINGS, profile.name, 'profile.csv'))

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
