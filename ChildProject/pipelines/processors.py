from abc import ABC, abstractmethod
import argparse
import datetime
import multiprocessing as mp
import numpy as np
import os, glob
import sys
import pandas as pd
import shutil
import subprocess
from typing import Union, List
from yaml import dump
import logging

from ..projects import ChildProject, CONVERTED_RECORDINGS
from .pipeline import Pipeline

from ChildProject import __version__

# Create a logger for the module (file)
logger_annotations = logging.getLogger(__name__)
# messages are propagated to the higher level logger (ChildProject), used in cmdline.py
logger_annotations.propagate = True

pipelines = {}


class AudioProcessor(ABC):
    def __init__(
        self,
        project: ChildProject,
        name: str,
        input_profile: str = None,
        threads: int = 1,
        recordings: Union[str, List[str], pd.DataFrame] = None,
    ):

        self.project = project
        self.name = name
        self.threads = int(threads)
        self.recordings = Pipeline.recordings_from_list(recordings)

        self.input_profile = input_profile

        if self.input_profile:
            input_path = os.path.join(
                self.project.path,
                CONVERTED_RECORDINGS,
                self.input_profile,
            )

            assert os.path.exists(
                input_path
            ), f"provided input profile {input_profile} does not exist"

        self.converted = pd.DataFrame()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        pipelines[cls.SUBCOMMAND] = cls

    def output_directory(self):
        return os.path.join(
            self.project.path,
            CONVERTED_RECORDINGS,
            self.name,
        )

    def read_metadata(self):
        path = os.path.join(self.output_directory(), "recordings.csv")

        if os.path.exists(path):
            return pd.read_csv(path).set_index(
                ["original_filename", "converted_filename"]
            )
        else:
            return None

    def export_metadata(self):
        path = os.path.join(self.output_directory(), "recordings.csv")

        self.converted.to_csv(path)

    @abstractmethod
    def process_recording(self, recording):
        pass

    def process(self, parameters):
        recordings = self.project.get_recordings_from_list(self.recordings)

        os.makedirs(name=self.output_directory(), exist_ok=True)

        if self.threads == 1:
            self.converted = [
                self.process_recording(recording)
                for recording in recordings.to_dict("records")
            ]
        else:
            with mp.Pool(processes=self.threads) as pool:
                self.converted = pool.map(
                    self.process_recording, recordings.to_dict("records")
                )

        previously_converted = self.read_metadata()
        self.converted = pd.concat(self.converted)

        if not len(self.converted):
            return

        self.converted.set_index(
            ["original_filename", "converted_filename"], inplace=True
        )
        self.converted = self.converted.assign(parameters=parameters)

        if previously_converted is not None:
            self.converted = pd.concat(
                [
                    previously_converted[
                        ~previously_converted.index.isin(self.converted.index)
                    ],
                    self.converted,
                ]
            )

        self.export_metadata()

    @staticmethod
    def add_parser(parsers):
        pass


class BasicProcessor(AudioProcessor):
    SUBCOMMAND = "basic"

    def __init__(
        self,
        project: ChildProject,
        name: str,
        format: str,
        codec: str,
        sampling: int,
        threads: int = 1,
        recordings: Union[str, List[str], pd.DataFrame] = None,
        skip_existing: bool = False,
        input_profile: str = None,
    ):

        super().__init__(
            project,
            name,
            threads=threads,
            recordings=recordings,
            input_profile=input_profile,
        )

        self.format = format
        self.codec = str(codec)
        self.sampling = int(sampling)
        self.skip_existing = bool(skip_existing)

    def process_recording(self, recording):
        if recording["recording_filename"] == "NA":
            return pd.DataFrame()

        original_file = self.project.get_recording_path(
            recording["recording_filename"], self.input_profile
        )

        destination_file = os.path.join(
            self.output_directory(),
            os.path.splitext(recording["recording_filename"])[0]
            + "."
            + self.format,
        )

        os.makedirs(name=os.path.dirname(destination_file), exist_ok=True)

        skip = self.skip_existing and (
            os.path.exists(destination_file) or os.path.islink(destination_file)
        )

        if skip:
            return pd.DataFrame()

        args = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            original_file,
            "-c:a",
            self.codec,
            "-ar",
            str(self.sampling),
        ]

        args.append(destination_file)

        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        (stdout, stderr) = proc.communicate()
        success = proc.returncode == 0

        if not success:
            logger_annotations.error(stderr)
            return pd.DataFrame(
                [
                    {
                        "original_filename": recording["recording_filename"],
                        "converted_filename": "",
                        "success": False,
                        "error": stderr,
                    }
                ]
            )
        else:
            converted_files = [
                os.path.splitext(recording["recording_filename"])[0]
                + "."
                + self.format
            ]

        return pd.DataFrame(
            [
                {
                    "original_filename": recording["recording_filename"],
                    "converted_filename": cf,
                    "success": True,
                }
                for cf in converted_files
            ]
        )

    @staticmethod
    def add_parser(subparsers, subcommand):
        parser = subparsers.add_parser(subcommand, help="basic audio conversion")
        parser.add_argument("name", help="name of the export profile")

        parser.add_argument("--format", help="audio format (e.g. wav)", required=True)
        parser.add_argument(
            "--codec", help="audio codec (e.g. pcm_s16le)", required=True
        )
        parser.add_argument(
            "--sampling",
            help="sampling frequency (e.g. 16000)",
            required=True,
            type=int,
        )
        
        parser.add_argument(
            "--skip-existing",
            dest="skip_existing",
            required=False,
            default=False,
            action="store_true",
        )
        parser.add_argument(
            "--recordings",
            help="list of recordings to process, separated by whitespaces; only values of 'recording_filename' present in the metadata are supported.",
            default=None,
            nargs="+",
        )


class VettingProcessor(AudioProcessor):
    SUBCOMMAND = "vetting"

    def __init__(
        self,
        project: ChildProject,
        name: str,
        segments_path: str,
        threads: int = 1,
        recordings: Union[str, List[str], pd.DataFrame] = None,
        input_profile: str = None,
    ):

        super().__init__(
            project,
            name,
            threads=threads,
            recordings=recordings,
            input_profile=input_profile,
        )
        self.segments = pd.read_csv(segments_path)

    def process_recording(self, recording):
        import librosa
        import soundfile

        original_file = self.project.get_recording_path(
            recording["recording_filename"], self.input_profile
        )

        destination_file = os.path.join(
            self.output_directory(),
            os.path.splitext(recording["recording_filename"])[0] + ".wav",
        )

        os.makedirs(name=os.path.dirname(destination_file), exist_ok=True)

        vettoed_segments = self.segments[
            self.segments["recording_filename"] == recording["recording_filename"]
        ]

        signal, sr = librosa.load(original_file, sr=None, mono=False)

        onsets = librosa.time_to_samples(
            times=vettoed_segments["segment_onset"].values / 1000, sr=sr
        )
        offsets = librosa.time_to_samples(
            times=vettoed_segments["segment_offset"].values / 1000, sr=sr
        )

        if signal.ndim == 1:
            for i in range(len(onsets)):
                signal[onsets[i] : offsets[i]] = 0

            soundfile.write(destination_file, signal, samplerate=sr)
        else:
            for i in range(len(onsets)):
                signal[:, onsets[i] : offsets[i]] = 0

            soundfile.write(destination_file, np.transpose(signal), samplerate=sr)

        return pd.DataFrame(
            [
                {
                    "original_filename": recording["recording_filename"],
                    "converted_filename": os.path.splitext(
                        recording["recording_filename"]
                    )[0]
                    + ".wav",
                    "success": True,
                }
            ]
        )

    @staticmethod
    def add_parser(subparsers, subcommand):
        parser = subparsers.add_parser(subcommand, help="vetting")
        parser.add_argument("name", help="name of the export profile")

        parser.add_argument(
            "--segments-path",
            help="path to the CSV dataframe containing the segments to be vetted",
            required=True,
        )
        parser.add_argument(
            "--recordings",
            help="list of recordings to process, separated by commas; only values of 'recording_filename' present in the metadata are supported.",
            default=None,
            nargs="+",
        )


class ChannelMapper(AudioProcessor):
    SUBCOMMAND = "channel-mapping"

    def __init__(
        self,
        project: ChildProject,
        name: str,
        channels: list,
        threads: int = 1,
        recordings: Union[str, List[str], pd.DataFrame] = None,
        input_profile: str = None,
    ):

        super().__init__(
            project,
            name,
            threads=threads,
            recordings=recordings,
            input_profile=input_profile,
        )

        self.channels = [list(map(float, channel.split(","))) for channel in channels]

        self.channels = np.array(self.channels)

    def process_recording(self, recording):
        import librosa
        import soundfile

        original_file = self.project.get_recording_path(
            recording["recording_filename"], self.input_profile
        )

        destination_file = os.path.join(
            self.output_directory(),
            os.path.splitext(recording["recording_filename"])[0] + ".wav",
        )

        os.makedirs(name=os.path.dirname(destination_file), exist_ok=True)

        df = pd.DataFrame(
            [
                {
                    "original_filename": recording["recording_filename"],
                    "converted_filename": os.path.splitext(
                        recording["recording_filename"]
                    )[0]
                    + ".wav",
                }
            ]
        )

        signal, sr = librosa.load(original_file, sr=None, mono=False)

        if self.channels.shape[1] != signal.shape[0]:
            logger_annotations.error(
                "skipping '%s' due to channel mismatch (expected %d channels, got %d)", 
                recording["recording_filename"], 
                self.channels.shape[1], 
                signal.shape[0],
                )
            
            return df.assign(success=False)

        output = np.matmul(self.channels, signal)
        soundfile.write(destination_file, np.transpose(output), samplerate=sr)

        return df.assign(success=True)

    @staticmethod
    def add_parser(subparsers, subcommand):
        parser = subparsers.add_parser(subcommand, help="channel mapping")
        parser.add_argument("name", help="name of the export profile")

        parser.add_argument(
            "--channels",
            help="lists of weigths for each channel",
            nargs="+",
            required=True,
        )
        parser.add_argument(
            "--recordings",
            help="list of recordings to process, separated by commas; only values of 'recording_filename' present in the metadata are supported.",
            default=None,
            nargs="+",
        )

class AudioStandard(AudioProcessor):
    SUBCOMMAND = "standard"

    def __init__(
        self,
        project: ChildProject,
        threads: int = 1,
        recordings: Union[str, List[str], pd.DataFrame] = None,
        skip_existing: bool = False,
        input_profile: str = None,
    ):

        super().__init__(
            project,
            name='standard',
            threads=threads,
            recordings=recordings,
            input_profile=input_profile,
        )

        self.format = "wav"
        self.codec = "pcm_s16le"
        self.sampling = "16000"
        self.skip_existing = bool(skip_existing)

    def process_recording(self, recording):
        if recording["recording_filename"] == "NA":
            return pd.DataFrame()

        original_file = self.project.get_recording_path(
            recording["recording_filename"], self.input_profile
        )

        destination_file = os.path.join(
            self.output_directory(),
            os.path.splitext(recording["recording_filename"])[0]
            + "."
            + self.format,
        )

        os.makedirs(name=os.path.dirname(destination_file), exist_ok=True)

        skip = self.skip_existing and (
            os.path.exists(destination_file) or os.path.islink(destination_file)
        )

        if skip:
            return pd.DataFrame()

        args = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            original_file,
            "-c:a",
            self.codec,
            "-ar",
            str(self.sampling),
            "-af",
            'pan=mono| c0=c0',
        ]

        args.append(destination_file)

        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        (stdout, stderr) = proc.communicate()
        success = proc.returncode == 0

        if not success:
            print(stderr, file=sys.stderr)

            return pd.DataFrame(
                [
                    {
                        "original_filename": recording["recording_filename"],
                        "converted_filename": "",
                        "success": False,
                        "error": stderr,
                    }
                ]
            )
        else:
            converted_files = [
                os.path.splitext(recording["recording_filename"])[0]
                + "."
                + self.format
            ]

        return pd.DataFrame(
            [
                {
                    "original_filename": recording["recording_filename"],
                    "converted_filename": cf,
                    "success": True,
                }
                for cf in converted_files
            ]
        )

    @staticmethod
    def add_parser(subparsers, subcommand):
        parser = subparsers.add_parser(subcommand, help="standard audio conversion")
        
        parser.add_argument(
            "--skip-existing",
            dest="skip_existing",
            required=False,
            default=False,
            action="store_true",
        )
        parser.add_argument(
            "--recordings",
            help="list of recordings to process, separated by whitespaces; only values of 'recording_filename' present in the metadata are supported.",
            default=None,
            nargs="+",
        )

class AudioProcessingPipeline(Pipeline):
    def __init__(self):
        pass

    def run(
        self,
        path: str,
        processor: str,
        threads: int = 1,
        func=None,
        **kwargs,
    ):
        parameters = locals()
        parameters = [
            {key: parameters[key]}
            for key in parameters
            if key not in ["self", "kwargs"]
        ]
        parameters.extend([{key: kwargs[key]} for key in kwargs])

        date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.project = ChildProject(path)
        self.project.read()

        if processor not in pipelines:
            raise NotImplementedError(f"invalid pipeline '{processor}'")

        proc = pipelines[processor](self.project, threads=threads, **kwargs)
        proc.process(f"parameters_{date}.yml")
     
        logger_annotations.info(
                "Exported audio to %s", 
                proc.output_directory(), 
                )
            
        

        parameters_path = os.path.join(
            proc.output_directory(), f"parameters_{date}.yml"
        )
        dump(
            {
                "parameters": parameters,
                "package_version": __version__,
                "date": date,
            },
            open(parameters_path, "w+"),
        )
        
        logger_annotations.info(
            "exported processor parameters to ",
            parameters_path,
            )   
        
        return os.path.join(proc.output_directory(), "recordings.csv"), parameters_path

    @staticmethod
    def setup_parser(parser):
        parser.add_argument("path", help="path to the dataset")
        #parser.add_argument("name", help="name of the export profile")

        subparsers = parser.add_subparsers(help="processor", dest="processor")
        for pipeline in pipelines:
            pipelines[pipeline].add_parser(subparsers, pipeline)

        parser.add_argument(
            "--threads",
            help="amount of threads running conversions in parallel (0 = uses all available cores)",
            required=False,
            default=1,
            type=int,
        )

        parser.add_argument(
            "--input-profile",
            help="profile of input recordings (process raw recordings by default)",
            default=None,
        )
