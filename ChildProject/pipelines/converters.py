from abc import ABC, abstractmethod
import argparse
import datetime
import multiprocessing as mp
import numpy as np
import operator
import os
import sys
import pandas as pd
from pydub import AudioSegment
import shutil
import subprocess
from yaml import dump

import ChildProject
from ChildProject.pipelines.pipeline import Pipeline

class AudioConverter(ABC):
    def __init__(self, project: ChildProject.projects.ChildProject, name: str):
        self.project = project
        self.name = name

        self.converted = []

    def export_metadata(self):
        destination = os.path.join(
            self.project.path,
            ChildProject.projects.ChildProject.CONVERTED_RECORDINGS,
            self.name,
            'recordings.csv'
        )

        pd.DataFrame(self.converted)\
            .set_index('converted_filename')\
            .to_csv(destination)


    @abstractmethod
    def process_recording(self, recording):
        pass

    @abstractmethod
    def process(self):
        pass

    @staticmethod
    def add_parser(parsers):
        pass

class DefaultConverter(AudioConverter):
    def __init__(self,
        project: ChildProject.projects.ChildProject,
        name: str,
        format: str,
        codec: str,
        sampling: int,
        split: str = None
        ):

        super().__init__(project, name)

        self.profile = RecordingProfile(
            name,
            format = format,
            codec = codec,
            sampling = sampling,
            split = split,
            extra_flags = None
        )

    def process_recording(self, recording):
        if recording['recording_filename'] == 'NA':
            return []

            original_file = os.path.join(
                self.project.path,
                ChildProject.RAW_RECORDINGS,
                self.recording['recording_filename']
            )

            destination_file = os.path.join(
                self.project.path,
                ChildProject.CONVERTED_RECORDINGS,
                self.profile.name,
                os.path.splitext(recording['recording_filename'])[0] + '.%03d.' + self.profile.format if self.profile.split
                else os.path.splitext(recording['recording_filename'])[0] + '.' + self.profile.format
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
                    '-c:a', self.profile.codec,
                    '-ar', str(self.profile.sampling)
                ]

                if self.profile.split:
                    args.extend([
                        '-segment_time', self.profile.split, '-f', 'segment'
                    ])

                args.append(destination_file)

                proc = subprocess.Popen(args, stdout = subprocess.DEVNULL, stderr = subprocess.PIPE)
                (stdout, stderr) = proc.communicate()
                success = proc.returncode == 0

            if not success:
                return [{
                    'original_filename': recording['recording_filename'],
                    'converted_filename': "",
                    'success': False,
                    'error': stderr
                }]
            else:
                if self.profile.split:
                    converted_files = [
                        os.path.basename(cf)
                        for cf in glob.glob(os.path.join(path, ChildProject.CONVERTED_RECORDINGS, self.profile.name, os.path.splitext(recording['recording_filename'])[0] + '.*.' + self.profile.format))
                    ]
                else:
                    converted_files = [os.path.splitext(recording['recording_filename'])[0] + '.' + self.profile.format]

            return [{
                'original_filename': recording['recording_filename'],
                'converted_filename': cf,
                'success': True
            } for cf in converted_files]
            
    def process(self):
        os.makedirs(
            name = os.path.join(self.project.path, ChildProject.CONVERTED_RECORDINGS, self.profile.name),
            exist_ok = True
        )

        conversion_table = []
        pool = mp.Pool(processes = self.threads if self.threads > 0 else mp.cpu_count())

        self.converted = pool.map(
            partial(convert_recording, self.project.path, self.profile, skip_existing),
            self.project.recordings.to_dict('records')
        )

        self.converted = reduce(operator.concat, conversion_table)
        self.export_metadata()
        pass

    @staticmethod
    def add_parser(parsers):
        pass

    @staticmethod
    def add_parser(samplers):
        parser = samplers.add_parser('ffmpeg', help = 'ffmpeg converter')

class FFMPEGConverter(AudioConverter):
    def __init__(self,
        project: ChildProject.projects.ChildProject,
        name: str
        ):

        super().__init__(project, name)

    def process_recording(self, recording):
        pass

    def process(self):
        pass

    @staticmethod
    def add_parser(parsers):
        pass

    @staticmethod
    def add_parser(samplers):
        parser = samplers.add_parser('ffmpeg', help = 'ffmpeg converter')

class VettingConverter(AudioConverter):
    def __init__(self,
        project: ChildProject.projects.ChildProject,
        name: str,
        segments_path: str
        ):

        super().__init__(project, name)
        self.segments = pd.read_csv(segments_path)

    def process_recording(self, recording):
        pass

    def process(self):
        pass

    @staticmethod
    def add_parser(parsers):
        pass

    @staticmethod
    def add_parser(samplers):
        parser = samplers.add_parser('ffmpeg', help = 'ffmpeg converter')

class AudioConverterPipeline(Pipeline):
    def __init__(self):
        pass

    def run(self, path: str, name: str, converter: str, skip_existing: bool = False, threads: int = 0, **kwargs):
        parameters = locals()
        parameters = [{key: parameters[key]} for key in parameters if key not in ['self', 'kwargs']]
        parameters.extend([{key: kwargs[key]} for key in kwargs])

        self.project = ChildProject.projects.ChildProject(path)
        self.project.read()

        cnvrtr = None
        if converter == 'default':
            cnvrtr = FFMPEGConverter(self.project, name, **kwargs)
        elif converter == 'ffmpeg':
            cnvrtr = FFMPEGConverter(self.project, name, **kwargs)

        if cnvrtr is None:
            raise Exception('invalid converter')

        cnvrtr.process()

        date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        os.makedirs(destination, exist_ok = True)
        segments_path = os.path.join(destination, 'segments_{}.csv'.format(date))
        parameters_path = os.path.join(destination, 'parameters_{}.yml'.format(date))

        self.segments[set(self.segments.columns) & {'recording_filename', 'segment_onset', 'segment_offset'}].to_csv(segments_path, index = False)
        print("exported audio to {}".format(segments_path))
        dump({
            'parameters': parameters,
            'package_version': ChildProject.__version__,
            'date': date
        }, open(parameters_path, 'w+'))
        print("exported sampler parameters to {}".format(parameters_path))

        return segments_path

    @staticmethod
    def setup_parser(parser):
        parser.add_argument('path', help = 'path to the dataset')
        parser.add_argument('name', help = 'name of the export profile')

        converters = parser.add_subparsers(help = 'converter', dest = 'converter')
        FFMPEGConverter.add_parser(converters)

