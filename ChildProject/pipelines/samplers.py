from abc import ABC, abstractmethod
import argparse
import datetime
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from pydub import AudioSegment
import sys
import traceback
from typing import Union, List
from yaml import dump
import logging

from ..projects import ChildProject
from ..annotations import AnnotationManager
from .pipeline import Pipeline

from ChildProject import __version__

# Create a logger for the module (file)
logger_annotations = logging.getLogger(__name__)
# messages are propagated to the higher level logger (ChildProject), used in cmdline.py
logger_annotations.propagate = True

pipelines = {}


class Sampler(ABC):
    def __init__(
        self,
        project: ChildProject,
        recordings: Union[str, List[str], pd.DataFrame] = None,
        exclude: Union[str, pd.DataFrame] = None,
    ):

        self.project = project

        self.segments = pd.DataFrame()
        self.annotation_set = ""
        self.target_speaker_type = []

        self.recordings = Pipeline.recordings_from_list(recordings)

        if exclude is None:
            self.excluded = pd.DataFrame(
                columns=["recording_filename", "segment_onset", "segment_offset"]
            )
        else:
            if not isinstance(exclude, pd.DataFrame):
                exclude = pd.read_csv(exclude)

            if not {"recording_filename", "segment_onset", "segment_offset"}.issubset(
                set(exclude.columns)
            ):
                raise ValueError(
                    "exclude dataframe is missing a 'recording_filename' column"
                )

            self.excluded = exclude

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        pipelines[cls.SUBCOMMAND] = cls

    @abstractmethod
    def _sample(self):
        pass

    @staticmethod
    @abstractmethod
    def add_parser(parsers):
        pass

    def sample(self):
        self._sample()
        self.remove_excluded()
        return self.segments

    def retrieve_segments(self, recording_filename=None):
        am = AnnotationManager(self.project)
        annotations = am.annotations
        annotations = annotations[annotations["set"] == self.annotation_set]

        if recording_filename:
            annotations = annotations[
                annotations["recording_filename"] == recording_filename
            ]

        if annotations.shape[0] == 0:
            return None

        try:
            segments = am.get_segments(annotations)
        except:
            return None

        if len(self.target_speaker_type) and len(segments):
            segments = segments[segments["speaker_type"].isin(self.target_speaker_type)]

        return segments

    def remove_excluded(self):
        if len(self.excluded) == 0:
            return

        from pyannote.core import Segment, Timeline

        segments = []
        for recording, _segments in self.segments.groupby("recording_filename"):
            sampled = Timeline(
                segments=[
                    Segment(segment_onset, segment_offset)
                    for segment_onset, segment_offset in _segments[
                        ["segment_onset", "segment_offset"]
                    ].values
                ]
            )

            excl_segments = self.excluded.loc[
                self.excluded["recording_filename"] == recording
            ]
            excl = Timeline(
                segments=[
                    Segment(segment_onset, segment_offset)
                    for segment_onset, segment_offset in excl_segments[
                        ["segment_onset", "segment_offset"]
                    ].values
                ]
            )

            # sampled = sampled.extrude(sampled) # not released yet
            extent_tl = Timeline([sampled.extent()], uri=sampled.uri)
            truncating_support = excl.gaps(support=extent_tl)
            sampled = sampled.crop(truncating_support, mode="intersection")

            segments.append(
                pd.DataFrame(
                    [[recording, s.start, s.end] for s in sampled],
                    columns=["recording_filename", "segment_onset", "segment_offset"],
                )
            )

        self.segments = pd.concat(segments)

    def assert_valid(self):
        require_columns = ["recording_filename", "segment_onset", "segment_offset"]
        missing_columns = list(set(require_columns) - set(self.segments.columns))

        if missing_columns:
            raise Exception(
                "custom segments are missing the following columns: {}".format(
                    ",".join(missing_columns)
                )
            )

    def export_audio(self, destination, profile=None, **kwargs):
        self.assert_valid()

        for recording, segments in self.segments.groupby("recording_filename"):
            path = self.project.get_recording_path(recording, profile)

            audio = AudioSegment.from_file(path)

            for segment in segments.to_dict(orient="records"):
                output_name = "{}_{}_{}.{}".format(
                    os.path.splitext(recording)[0],
                    segment["segment_onset"],
                    segment["segment_offset"],
                    "wav",
                )
                output_path = os.path.join(destination, output_name)
                seg = audio[segment["segment_onset"] : segment["segment_offset"]]

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                seg.export(output_path, **kwargs)


class CustomSampler(Sampler):
    SUBCOMMAND = "custom"

    def __init__(
        self,
        project: ChildProject,
        segments_path: str,
        recordings: Union[str, List[str], pd.DataFrame] = None,
        exclude: Union[str, pd.DataFrame] = None,
    ):

        super().__init__(project, recordings, exclude)
        self.segments_path = segments_path

    def _sample(self: str):
        self.segments = pd.read_csv(self.segments_path)
        return self.segments

    @staticmethod
    def add_parser(subparsers, subcommand):
        parser = subparsers.add_parser(subcommand, help="custom sampling")
        parser.add_argument("segments", help="path to selected segments datafame")


class PeriodicSampler(Sampler):
    """Periodic sampling of a recording.

    :param project: ChildProject instance of the target dataset.
    :type project: ChildProject.projects.ChildProject
    :param length: length of each segment, in milliseconds
    :type length: int
    :param period: spacing between two consecutive segments, in milliseconds
    :type period: int
    :param offset: offset of the first segment, in milliseconds, defaults to 0
    :type offset: int
    :param recordings: recordings to sample from; if None, all recordings will be sampled, defaults to None
    :type recordings: Union[str, List[str], pd.DataFrame], optional
    """

    SUBCOMMAND = "periodic"

    def __init__(
        self,
        project: ChildProject,
        length: int,
        period: int,
        offset: int = 0,
        profile: str = None,
        recordings: Union[str, List[str], pd.DataFrame] = None,
        exclude: Union[str, pd.DataFrame] = None,
    ):

        super().__init__(project, recordings, exclude)
        self.length = int(length)
        self.period = int(period)
        self.offset = int(offset)
        self.profile = profile

    def _sample(self):
        recordings = self.project.get_recordings_from_list(self.recordings)

        if not "duration" in recordings.columns:
            logger_annotations.error("recordings duration was not found in the metadata and an attempt will be made to calculate it.")

            durations = self.project.compute_recordings_duration(self.profile).dropna()
            recordings = recordings.merge(
                durations[durations["recording_filename"] != "NA"],
                how="left",
                left_on="recording_filename",
                right_on="recording_filename",
            )

        recordings["duration"].fillna(0, inplace=True)

        self.segments = recordings[["recording_filename", "duration"]].copy()
        self.segments["segment_onset"] = self.segments.apply(
            lambda row: np.arange(
                self.offset,
                row["duration"] - self.length + 1e-4,
                self.period + self.length,
            ),
            axis=1,
        )
        self.segments = self.segments.explode("segment_onset")
        # discard recordings that can't include segments (they are NA here bc explode keeps empty lists)
        self.segments = self.segments.dropna(subset=['segment_onset'])
        self.segments["segment_onset"] = self.segments["segment_onset"].astype(int)
        self.segments["segment_offset"] = self.segments["segment_onset"] + self.length
        self.segments.rename(
            columns={"recording_filename": "recording_filename"}, inplace=True
        )

        return self.segments

    @staticmethod
    def add_parser(subparsers, subcommand):
        parser = subparsers.add_parser(subcommand, help="periodic sampling")
        parser.add_argument(
            "--length",
            help="length of each segment, in milliseconds",
            type=float,
            required=True,
        )
        parser.add_argument(
            "--period",
            help="spacing between two consecutive segments, in milliseconds",
            type=float,
            required=True,
        )
        parser.add_argument(
            "--offset",
            help="offset of the first segment, in milliseconds",
            type=float,
            default=0,
        )
        parser.add_argument(
            "--profile",
            help="name of the profile of recordings to use to estimate duration (uses raw recordings if empty)",
            default="",
            type=str,
        )


class RandomVocalizationSampler(Sampler):
    """Sample vocalizations based on some input annotation set.

    :param project: ChildProject instance of the target dataset.
    :type project: ChildProject.projects.ChildProject
    :param annotation_set: Set of annotations to get vocalizations from.
    :type annotation_set: str
    :param target_speaker_type: List of speaker types to sample vocalizations from.
    :type target_speaker_type: list
    :param sample_size: Amount of vocalizations to sample, per recording.
    :type sample_size: int
    :param by: units to sample from, defaults to 'recording_filename'
    :type by: str, optional
    :param recordings: recordings to sample from; if None, all recordings will be sampled, defaults to None
    :type recordings: Union[str, List[str], pd.DataFrame], optional
    :param threads: amount of threads to run on, defaults to 1
    :type threads: int, optional
    """

    SUBCOMMAND = "random-vocalizations"

    def __init__(
        self,
        project: ChildProject,
        annotation_set: str,
        target_speaker_type: list,
        sample_size: int,
        threads: int = 1,
        by: str = "recording_filename",
        recordings: Union[str, List[str], pd.DataFrame] = None,
        exclude: Union[str, pd.DataFrame] = None,
    ):

        super().__init__(project, recordings, exclude)
        self.annotation_set = annotation_set
        self.target_speaker_type = target_speaker_type
        self.sample_size = sample_size
        self.threads = threads
        self.by = by

    def _get_segments(self, recording):
        segments = self.retrieve_segments(recording["recording_filename"])

        if segments is None:
            logger_annotations.warning(
                "no annotations from the set '%s' were found for the recording '%s'", 
                self.annotation_set, 
                recording["recording_filename"], 
                )
            return pd.DataFrame(
                columns=["segment_onset", "segment_offset", "recording_filename"]
            )

        return segments

    def _sample_unit(self, group):
        unit, recordings = group
        recordings[self.by] = unit
        segments = pd.concat(
            [self._get_segments(r) for r in recordings.to_dict(orient="records")]
        )

        return segments.sample(frac=1).head(self.sample_size)

    def _sample(self):
        recordings = self.project.get_recordings_from_list(self.recordings)

        if self.threads == 1:
            self.segments = map(self._sample_unit, recordings.groupby(self.by))
        else:
            with mp.Pool(
                processes=self.threads if self.threads >= 1 else mp.cpu_count()
            ) as pool:
                self.segments = pool.map(self._sample_unit, recordings.groupby(self.by))

        self.segments = pd.concat(self.segments)

        return self.segments

    @staticmethod
    def add_parser(subparsers, subcommand):
        parser = subparsers.add_parser(subcommand, help="random sampling")
        parser.add_argument("--annotation-set", help="annotation set", default="vtc")
        parser.add_argument(
            "--target-speaker-type",
            help="speaker type to get chunks from",
            choices=["CHI", "OCH", "FEM", "MAL"],
            nargs="+",
            default=["CHI"],
        )
        parser.add_argument(
            "--sample-size",
            help="how many samples per unit (recording, session, or child)",
            required=True,
            type=int,
        )
        parser.add_argument(
            "--threads", help="amount of threads to run on", default=1, type=int
        )
        parser.add_argument(
            "--by",
            help="units to sample from (default behavior is to sample by recording)",
            choices=["recording_filename", "session_id", "child_id"],
            default="recording_filename",
        )


class EnergyDetectionSampler(Sampler):
    """Sample windows within each recording, targetting those
    that have a signal energy higher than some threshold.

    :param project: ChildProject instance of the target dataset.
    :type project: ChildProject.projects.ChildProject
    :param windows_length: Length of each window, in milliseconds.
    :type windows_length: int
    :param windows_spacing: Spacing between the start of each window, in milliseconds.
    :type windows_spacing: int
    :param windows_count: How many windows to retain per recording.
    :type windows_count: int
    :param windows_offset: start of the first window, in milliseconds, defaults to 0
    :type windows_offset: float, optional
    :param threshold: lowest energy quantile to sample from, defaults to 0.8
    :type threshold: float, optional
    :param low_freq: if > 0, frequencies below will be filtered before calculating the energy, defaults to 0
    :type low_freq: int, optional
    :param high_freq: if < 100000, frequencies above will be filtered before calculating the energy, defaults to 100000
    :type high_freq: int, optional
    :param by: units to sample from, defaults to 'recording_filename'
    :type by: str, optional
    :param recordings: recordings to sample from; if None, all recordings will be sampled, defaults to None
    :type recordings: Union[str, List[str], pd.DataFrame], optional
    :param threads: amount of threads to run on, defaults to 1
    :type threads: int, optional
    """

    SUBCOMMAND = "energy-detection"

    def __init__(
        self,
        project: ChildProject,
        windows_length: int,
        windows_spacing: int,
        windows_count: int,
        windows_offset: int = 0,
        threshold: float = 0.8,
        low_freq: int = 0,
        high_freq: int = 100000,
        threads: int = 1,
        profile: str = "",
        by: str = "recording_filename",
        recordings: Union[str, List[str], pd.DataFrame] = None,
        exclude: Union[str, pd.DataFrame] = None,
    ):

        super().__init__(project, recordings, exclude)
        self.windows_length = int(windows_length)
        self.windows_count = int(windows_count)
        self.windows_spacing = int(windows_spacing)
        self.windows_offset = int(windows_offset)
        self.threshold = threshold
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.threads = threads
        self.profile = profile
        self.by = by

    def compute_energy_loudness(self, chunk, sampling_frequency: int):
        if self.low_freq > 0 or self.high_freq < 100000:
            chunk_fft = np.fft.fft(chunk)
            freq = np.abs(np.fft.fftfreq(len(chunk_fft), 1 / sampling_frequency))
            chunk_fft = chunk_fft[(freq > self.low_freq) & (freq < self.high_freq)]
            return np.sum(np.abs(chunk_fft) ** 2) / len(chunk)
        else:
            return np.sum(chunk ** 2)

    def get_recording_windows(self, recording):
        recording_path = self.project.get_recording_path(
            recording["recording_filename"], self.profile
        )

        if recording_path is None:
            logger_annotations.error(
                "failed to retrieve the path to '%s' (profile: %s)", 
                recording["recording_filename"], 
                self.profile, 
                )
            return pd.DataFrame()

        try:
            audio = AudioSegment.from_file(recording_path)
        except:
            logger_annotations.error(
                "failed to read '%s', is it a valid audio file ? %s", 
                recording_path, 
                traceback.format_exc(), 
                )
            return pd.DataFrame()

        duration = int(audio.duration_seconds * 1000)
        channels = audio.channels
        frequency = int(audio.frame_rate)
        max_value = 256 ** (int(audio.sample_width)) / 2 - 1

        windows_starts = np.arange(
            self.windows_offset, duration - self.windows_length, self.windows_spacing
        ).astype(int)
        windows = []
        logger_annotations.info(
                "computing the energy of %d windows for recording %s...", 
                len(windows_starts), 
                recording["recording_filename"], 
                )
        for start in windows_starts:
            energy = 0
            chunk = audio[start : start + self.windows_length].get_array_of_samples()
            channel_energies = np.zeros(channels)

            for channel in range(channels):
                data = np.array(chunk[channel::channels]) / max_value
                channel_energies[channel] = self.compute_energy_loudness(
                    data, frequency
                )

            window = {
                "segment_onset": start,
                "segment_offset": start + self.windows_length,
                "recording_filename": recording["recording_filename"],
                "energy": np.sum(channel_energies),
            }

            window[self.by] = str(recording[self.by])

            window.update(
                {
                    "channel_{}".format(channel): channel_energies[channel]
                    for channel in range(channels)
                }
            )
            windows.append(window)

        return pd.DataFrame(windows)

    def _sample(self):
        recordings = self.project.get_recordings_from_list(self.recordings)

        if self.threads == 1:
            windows = pd.concat(
                [
                    self.get_recording_windows(r)
                    for r in recordings.to_dict(orient="records")
                ]
            )
        else:
            with mp.Pool(
                processes=self.threads if self.threads >= 1 else mp.cpu_count()
            ) as pool:
                windows = pd.concat(
                    pool.map(
                        self.get_recording_windows, recordings.to_dict(orient="records")
                    )
                )

        windows = windows.set_index(self.by).merge(
            windows.groupby(self.by).agg(
                energy_threshold=("energy", lambda a: np.quantile(a, self.threshold))
            ),
            left_index=True,
            right_index=True,
        )
        windows = windows[windows["energy"] >= windows["energy_threshold"]]

        self.segments = windows.sample(frac=1).groupby(self.by).head(self.windows_count)
        self.segments.reset_index(inplace=True)
        self.segments.drop_duplicates(
            ["recording_filename", "segment_onset", "segment_offset"], inplace=True
        )

    @staticmethod
    def add_parser(subparsers, subcommand):
        parser = subparsers.add_parser(
            subcommand, help="energy based activity detection"
        )
        parser.add_argument(
            "--windows-length",
            help="length of each window (in milliseconds)",
            required=True,
            type=int,
        )
        parser.add_argument(
            "--windows-spacing",
            help="spacing between the start of two consecutive windows (in milliseconds)",
            required=True,
            type=int,
        )
        parser.add_argument(
            "--windows-count",
            help="how many windows to sample from each unit (recording, session, or child)",
            required=True,
            type=int,
        )
        parser.add_argument(
            "--windows-offset",
            help="start of the first window (in milliseconds)",
            type=int,
            default=0,
        )
        parser.add_argument(
            "--threshold",
            help="lowest energy quantile to sample from. default is 0.8 (i.e., sample from the 20%% windows with the highest energy).",
            default=0.8,
            type=float,
        )
        parser.add_argument(
            "--low-freq",
            help="remove all frequencies below low-freq before calculating each window's energy. (in Hz)",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--high-freq",
            help="remove all frequencies above high-freq before calculating each window's energy. (in Hz)",
            default=100000,
            type=int,
        )
        parser.add_argument(
            "--threads", help="amount of threads to run on", default=1, type=int
        )
        parser.add_argument(
            "--profile",
            help="name of the profile of recordings to use (uses raw recordings if empty)",
            default="",
            type=str,
        )
        parser.add_argument(
            "--by",
            help="units to sample from (default behavior is to sample by recording)",
            choices=["recording_filename", "session_id", "child_id"],
            default="recording_filename",
        )


class HighVolubilitySampler(Sampler):
    """Return the top ``windows_count`` windows (of length ``windows_length``)
    with the highest volubility from each recording,
    as calculated from the metric ``metric``.

    ``metrics`` can be any of three values: words, turns, and vocs.

     - The **words** metric sums the amount of words within each window. For LENA annotations, it is equivalent to **awc**.
     - The **turns** metric (aka ctc) sums conversational turns within each window. It relies on **lena_conv_turn_type** for LENA annotations. For other annotations, turns are estimated as adult/child speech switches in close temporal proximity.
     - The **vocs** metric sums vocalizations within each window. If ``metric="vocs"`` and ``speakers=['CHI']``, it is equivalent to the usual cvc metric (child vocalization counts).

    :param project: ChildProject instance of the target dataset.
    :type project: ChildProject.projects.ChildProject
    :param annotation_set: set of annotations to calculate volubility from.
    :type annotation_set: str
    :param metric: the metric to evaluate high-volubility. should be any of 'words', 'turns', 'vocs'.
    :type metric: str
    :param windows_length: length of the windows, in milliseconds
    :type windows_length: int
    :param windows_count: amount of top regions to extract per recording
    :type windows_count: int
    :param by: units to sample from, defaults to 'recording_filename'
    :type by: str, optional
    :param recordings: recordings to sample from; if None, all recordings will be sampled, defaults to None
    :type recordings: Union[str, List[str], pd.DataFrame], optional
    :param threads: amount of threads to run the sampler on
    :type threads: int
    """

    SUBCOMMAND = "high-volubility"

    def __init__(
        self,
        project: ChildProject,
        annotation_set: str,
        metric: str,
        windows_length: int,
        windows_count: int,
        speakers: List[str] = ["FEM", "MAL", "CHI"],
        threads: int = 1,
        by: str = "recording_filename",
        recordings: Union[str, List[str], pd.DataFrame] = None,
        exclude: Union[str, pd.DataFrame] = None,
    ):

        super().__init__(project, recordings, exclude)
        self.annotation_set = annotation_set
        self.metric = metric
        self.windows_length = windows_length
        self.windows_count = windows_count
        self.speakers = speakers
        self.threads = threads
        self.by = by

    def _segment_scores(self, recording):
        segments = self.retrieve_segments(recording["recording_filename"])

        if segments is None:
            logger_annotations.warning(
                "no annotations from the set '%s' were found for the recording '%s'", 
                self.annotation_set, 
                recording["recording_filename"], 
                )
            return pd.DataFrame(
                columns=["segment_onset", "segment_offset", "recording_filename"]
            )

        # NOTE: The timestamps were simply rounded to 5 minutes in the original
        # code via a complicated string replacement , which imo was incorrect, but
        # there are edge cases that must be decided (even though they are quiet small)
        segments["chunk"] = (
            segments["segment_offset"] // self.windows_length + 1
        ).astype("int")

        segment_onsets = segments.groupby("chunk")["segment_onset"].min()
        segment_offsets = segments.groupby("chunk")["segment_offset"].max()

        # this dataframe contains the segment onset and offsets for the chunks we calculated.
        windows = pd.merge(
            segment_onsets, segment_offsets, left_index=True, right_index=True
        ).reset_index()
        windows["recording_filename"] = recording["recording_filename"]

        if self.metric == "turns":
            if "lena_conv_turn_type" in segments.columns:
                # NOTE: This is the equivalent of CTC (tab1) in rlena_extract.R
                segments["is_CT"] = segments["lena_conv_turn_type"].isin(
                    ["TIFR", "TIMR"]
                )
            else:
                segments = segments[segments["speaker_type"].isin(self.speakers)]
                segments["iti"] = segments["segment_onset"] - segments[
                    "segment_offset"
                ].shift(1)
                segments["prev_speaker_type"] = segments["speaker_type"].shift(1)

                key_child_environment = set(self.speakers) - {"CHI"}

                segments["is_CT"] = (segments["iti"] < 1000) & (
                    (
                        (segments["speaker_type"] == "CHI")
                        & (segments["prev_speaker_type"].isin(key_child_environment))
                    )
                    | (
                        (segments["speaker_type"].isin(key_child_environment))
                        & (segments["prev_speaker_type"] == "CHI")
                    )
                )

            segments = (
                segments.groupby("chunk", as_index=False)[["is_CT"]]
                .sum()
                .rename(columns={"is_CT": self.metric})
                .merge(windows)
            )

        elif self.metric == "vocs":
            # NOTE: This is the equivalent of CVC (tab2) in rlena_extract.R
            if "utterances_count" in segments.columns:
                segments = (
                    segments[segments.speaker_type.isin(self.speakers)]
                    .groupby("chunk", as_index=False)[["utterances_count"]]
                    .sum()
                    .rename(columns={"utterances_count": self.metric})
                    .merge(windows)
                )
            else:
                if 'speaker_type' in segments:
                    segments = (
                        segments[segments.speaker_type.isin(self.speakers)]
                        .groupby("chunk", as_index=False)[["segment_onset"]]
                        .count()
                        .rename(columns={"segment_onset": self.metric})
                        .merge(windows)
                    )
                else:
                    segments = pd.DataFrame()

        elif self.metric == "words":
            # NOTE: This is the equivalent of AWC (tab3) in rlena_extract.R
            segments = (
                segments[segments.speaker_type.isin(self.speakers)]
                .groupby("chunk", as_index=False)[["words"]]
                .sum()
                .merge(windows)
            )

        else:
            raise ValueError("unknown metric '{}'".format(self.metric))

        return segments

    def _sample_unit(self, group):
        unit, recordings = group
        recordings[self.by] = unit
        segments = pd.concat(
            [self._segment_scores(r) for r in recordings.to_dict(orient="records")]
        )

        return (
            segments.sort_values(self.metric, ascending=False)
            .head(self.windows_count)
            .reset_index(drop=True)
        )

    def _sample(self):
        recordings = self.project.get_recordings_from_list(self.recordings)

        if self.threads == 1:
            self.segments = map(self._sample_unit, recordings.groupby(self.by))
        else:
            with mp.Pool(
                processes=self.threads if self.threads >= 1 else mp.cpu_count()
            ) as pool:
                self.segments = pool.map(self._sample_unit, recordings.groupby(self.by))

        self.segments = pd.concat(self.segments)

    @staticmethod
    def add_parser(subparsers, subcommand):
        parser = subparsers.add_parser(
            subcommand, help="high-volubility targeted sampling"
        )
        parser.add_argument("--annotation-set", help="annotation set", required=True)
        parser.add_argument(
            "--metric",
            help="which metric should be used to evaluate volubility",
            required=True,
            choices=["turns", "vocs", "words"],
        )
        parser.add_argument(
            "--windows-length",
            help="window length (milliseconds)",
            required=True,
            type=int,
        )
        parser.add_argument(
            "--windows-count",
            help="how many windows to be sampled from each unit (recording, session, or child)",
            required=True,
            type=int,
        )
        parser.add_argument(
            "--speakers",
            help="speakers to include",
            default=["CHI", "FEM", "MAL"],
            nargs="+",
            choices=["CHI", "FEM", "MAL", "OCH"],
        )
        parser.add_argument(
            "--threads", help="amount of threads to run on", default=1, type=int
        )
        parser.add_argument(
            "--by",
            help="units to sample from (default behavior is to sample by recording)",
            choices=["recording_filename", "session_id", "child_id"],
            default="recording_filename",
        )


class ConversationSampler(Sampler):
    """Conversation sampler.

    :param project: ChildProject instance
    :type project: ChildProject.projects.ChildProject
    :param annotation_set: set of annotation to derive conversations from
    :type annotation_set: str
    :param count: amount of conversations to sample
    :type count: int
    :param interval: maximum time-interval between two consecutive vocalizations (in milliseconds) to consider them part of the same conversational block, defaults to 1000
    :type interval: int, optional
    :param speakers: list of speakers to target, defaults to ["FEM", "MAL", "CHI"]
    :type speakers: List[str], optional
    :param threads: threads to run on, defaults to 1
    :type threads: int, optional
    :param by: units to sample from, defaults to "recording_filename"
    :type by: str, optional
    :param recordings: whitelist of recordings, defaults to None
    :type recordings: Union[str, List[str], pd.DataFrame], optional
    :param exclude: portions to exclude, defaults to None
    :type exclude: Union[str, pd.DataFrame], optional
    """

    SUBCOMMAND = "conversations"

    def __init__(
        self,
        project: ChildProject,
        annotation_set: str,
        count: int,
        interval: int = 1000,
        speakers: List[str] = ["FEM", "MAL", "CHI"],
        threads: int = 1,
        by: str = "recording_filename",
        recordings: Union[str, List[str], pd.DataFrame] = None,
        exclude: Union[str, pd.DataFrame] = None,
    ):

        super().__init__(project, recordings, exclude)
        self.annotation_set = annotation_set
        self.interval = interval
        self.count = count
        self.speakers = speakers
        self.threads = threads
        self.by = by

    def _retrieve_conversations(self, recording):
        segments = self.retrieve_segments(recording["recording_filename"])

        if segments is None or "speaker_type" not in segments.columns:
            logger_annotations.warning(
                "no annotations from the set '%s' were found for the recording '%s' or speaker_type column is missing", 
                self.annotation_set, 
                recording["recording_filename"], 
                )
            return pd.DataFrame(
                columns=[
                    "segment_onset",
                    "segment_offset",
                    "recording_filename",
                    "turns",
                ]
            )

        segments = segments[segments["speaker_type"].isin(self.speakers)]

        segments["iti"] = segments["segment_onset"] - segments["segment_offset"].shift(
            1
        )
        segments["breaks_chain"] = segments["iti"] > self.interval

        segments["prev_speaker_type"] = segments["speaker_type"].shift(1)
        key_child_environment = set(self.speakers) - {"CHI"}

        segments["is_CT"] = (
            (segments["speaker_type"] == "CHI")
            & (segments["prev_speaker_type"].isin(key_child_environment))
        ) | (
            (segments["speaker_type"].isin(key_child_environment))
            & (segments["prev_speaker_type"] == "CHI")
        )

        conversations = segments.groupby(segments["breaks_chain"].cumsum()).agg(
            recording_filename=("recording_filename", "first"),
            segment_onset=("segment_onset", "first"),
            segment_offset=("segment_offset", "last"),
            turns=("is_CT", "sum"),
        )

        return conversations

    def _sample_unit(self, group):
        unit, recordings = group
        recordings[self.by] = unit
        conversations = pd.concat(
            [
                self._retrieve_conversations(r)
                for r in recordings.to_dict(orient="records")
            ]
        )

        return (
            conversations.sort_values("turns", ascending=False)
            .head(self.count)
            .reset_index(drop=True)
        )

    def _sample(self):
        recordings = self.project.get_recordings_from_list(self.recordings)

        if self.threads == 1:
            self.segments = map(self._sample_unit, recordings.groupby(self.by))
        else:
            with mp.Pool(
                processes=self.threads if self.threads >= 1 else mp.cpu_count()
            ) as pool:
                self.segments = pool.map(self._sample_unit, recordings.groupby(self.by))

        self.segments = pd.concat(self.segments)

    @staticmethod
    def add_parser(subparsers, subcommand):
        parser = subparsers.add_parser(subcommand, help="convesation sampler")
        parser.add_argument("--annotation-set", help="annotation set", required=True)
        parser.add_argument(
            "--count",
            help="how many conversations to be sampled from each unit (recording, session, or child)",
            required=True,
            type=int,
        )
        parser.add_argument(
            "--interval",
            help="maximum time-interval between two consecutive vocalizations (in milliseconds) to consider them to be part of the same conversational block. default is 1000",
            default=1000,
            type=int,
        )
        parser.add_argument(
            "--speakers",
            help="speakers to include",
            default=["CHI", "FEM", "MAL"],
            nargs="+",
            choices=["CHI", "FEM", "MAL", "OCH"],
        )
        parser.add_argument(
            "--threads", help="amount of threads to run on", default=1, type=int
        )
        parser.add_argument(
            "--by",
            help="units to sample from (default behavior is to sample by recording)",
            choices=["recording_filename", "session_id", "child_id"],
            default="recording_filename",
        )


class SamplerPipeline(Pipeline):
    def __init__(self):
        self.segments = []

    def run(self, path, destination, sampler, func=None, **kwargs):
        parameters = locals()
        parameters = [
            {key: parameters[key]}
            for key in parameters
            if key not in ["self", "kwargs"]
        ]
        parameters.extend([{key: kwargs[key]} for key in kwargs])

        self.project = ChildProject(path)
        self.project.read()

        if sampler not in pipelines:
            raise NotImplementedError(f"invalid pipeline '{sampler}'")

        splr = pipelines[sampler](self.project, **kwargs)

        splr.sample()
        splr.assert_valid()
        self.segments = splr.segments

        date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        os.makedirs(destination, exist_ok=True)
        segments_path = os.path.join(destination, "segments_{}.csv".format(date))
        parameters_path = os.path.join(destination, "parameters_{}.yml".format(date))

        self.segments[
            set(self.segments.columns)
            & {"recording_filename", "segment_onset", "segment_offset"}
        ].to_csv(segments_path, index=False)
        logger_annotations.info(
                "exported sampled segments to %s", 
                segments_path, 
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
                "exported sampler parameters to %s", 
                parameters_path, 
                )

        return segments_path

    @staticmethod
    def setup_parser(parser):
        parser.add_argument("path", help="path to the dataset")
        parser.add_argument("destination", help="segments destination")

        subparsers = parser.add_subparsers(help="sampler", dest="sampler")
        for pipeline in pipelines:
            pipelines[pipeline].add_parser(subparsers, pipeline)

        parser.add_argument(
            "--recordings",
            help="path to a CSV dataframe containing the list of recordings to sample from (by default, all recordings will be sampled). The CSV should have one column named recording_filename.",
            default=None,
        )

        parser.add_argument(
            "--exclude",
            help="path to a CSV dataframe containing the list of segments to exclude. The columns should be: recording_filename, segment_onset and segment_offset.",
            default=None,
        )

