from collections.abc import Callable
from math import ceil, floor
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import librosa
from typing import List

from ChildProject.projects import STANDARD_PROFILE, STANDARD_SAMPLE_RATE, ChildProject

class Derivator(ABC):
    """
    Class used by a derivation process to carry it out. It defines a derive method and a get_auto_metadata method
    """
    def __init__(self):
        pass

    @abstractmethod
    def derive(self,
               project: ChildProject,
               metadata: dict,
               segments: pd.DataFrame
               ) -> pd.DataFrame:
        """
        Modifies the content of segments given the current derivation (actual logic of the derivation)

        :param project: dataset the derivation is run on
        :type project: ChildProject
        :param metadata: dictionary of all the information on the element of the dataset being processed (recording, child etc)
        :type metadata: dict
        :params segments: annotation segment to derive
        :type segments: pd.DataFrame
        """
        pass

    @abstractmethod
    def get_auto_metadata(self,
                          am: 'AnnotationManager',
                          input_set: str,
                          output_set: str
                        ) -> dict:
        """
        Creates the set metadata that should be stored in the new set

        :param am: AnnotationManager object carrying out the derivation
        :type am: AnnotationManager
        :param input_set: set used as the input of the derivation
        :type input_set: str
        :param output_set: set to output to
        :type output_set: str
        """
        pass


class RuntimeDerivator(Derivator):
    """
    Derivator object using a standolone function/callable. the metadata returned is empty, no parameters
    """

    def __init__(self, func: Callable):
        self.function = func

    def derive(self,
               project: ChildProject,
               metadata: dict,
               segments: pd.DataFrame
               ) -> pd.DataFrame:
        return self.function(project, metadata, segments)

    def get_auto_metadata(self,
                          am: 'AnnotationManager',
                          input_set: str,
                          output_set: str
                          ) -> dict:
        return {}

class AcousticDerivator(Derivator):
    """
    Derivator for generating acoustics description of identified vocalizations

    Based on the existing segmentation, extracts acoustics features of each vocalization identified. In particular,
            mean pitch semitone, median pitch semitone as well as 5th and 95th percentile of pitch semitone.
    """
    def __init__(self,
                 profile: str = str(STANDARD_PROFILE),
                 target_sr: int = STANDARD_SAMPLE_RATE,
                 ):
        """
        :param profile: profile of audio conversaion to use in the dataset
        :type profile: str
        :param target_sr: sampling rate used to extract acoustic ffeatures
        :type target_sr: int
        """
        self.profile = profile
        self.target_sr = target_sr

        super().__init__()

    @staticmethod
    @np.vectorize
    def f2st(f_Hz, base=50) -> float:
        """
        Returns the semitone of the frequency given as input adapted from https://rdrr.io/cran/hqmisc/src/R/hqmisc.R
        itself adapted from http://ldc.upenn.edu/myl/llog/semitones.R (no longer available).
        See https://www.internationalphoneticassociation.org/icphs-proceedings/ICPhS2003/papers/p15_0771.pdf for reference
        :param f_Hz: frequency to convert (in Herz)
        :type f_Hz: int
        :param base: baseline frequency relative to which semitones are expressed
        :type base: int
        :return: semitone corresponding to the frequency given as input
        :rtype: float
        """
        # Use a more explicit designation in annotation title
        AcousticDerivator.f2st.__name__ = 'semitone'

        semi = np.log(2 ** (1 / 12))
        return (np.log(f_Hz) - np.log(base)) / semi

    @staticmethod
    def get_pitch(audio_time_series, sampling_rate, func=None) -> dict:
        """
        Returns pitch-related annotations.
        Regarding pitch range, we use the 5-th percentile as the bottom of the range, and the 95-th percentile as the top.
        (see https://www.ibm.com/docs/en/wvs/6.1.1?topic=guide-introduction-pitch-its-use-ssml or
        https://languagelog.ldc.upenn.edu/nll/?p=40788 who also use the same methodology)
        :param audio_time_series: real-valued vector
        :type audio_time_series: np.array
        :param sampling_rate: sampling rate
        :type sampling_rate: int
        :param func: transformation function to apply to the fundamental frequency
        :type func: callable
        :return: raw pitch, mean pitch, median pitch, 5-th percentile, 95-th percentile, pitch range
        :rtype: dict
        """
        f0 = librosa.yin(audio_time_series,
                         fmin=60,
                         fmax=500,
                         sr=sampling_rate)  # pyin does not work, why?
        pitch = func(f0) if callable(func) else f0
        mean_pitch, median_pitch, p5_pitch, p95_pitch = pitch.mean(), np.quantile(pitch, .5), \
                                                        np.percentile(pitch, 5), np.percentile(pitch, 95)

        pitch_type = "f0" if not callable(func) else func.__name__

        return {"raw_pitch_{}".format(pitch_type): f0,
                "mean_pitch_{}".format(pitch_type): mean_pitch,
                "median_pitch_{}".format(pitch_type): median_pitch,
                "p5_pitch_{}".format(pitch_type): p5_pitch,
                "p95_pitch_{}".format(pitch_type): p95_pitch,
                "pitch_range_{}".format(pitch_type): p95_pitch - p5_pitch}

    def derive(self,
               project: ChildProject,
               metadata: dict,
               segments: pd.DataFrame,
              ) -> pd.DataFrame:
        recording = project.get_recording_path(metadata['recording_filename'], profile=self.profile)
        file_sr = librosa.get_samplerate(recording)
        assert file_sr == self.target_sr, ValueError("Mismatch between file's true sampling rate ({}) and "
                                                "target sampling rate ({})!".format(file_sr, self.target_sr))
        #audio_time_series, sampling_rate = librosa.load(recording, mono=True, sr=self.target_sr)

        # Computes the start frame and end frame of the given segments given is on/offset in seconds
        segments['extended_onset'] = segments['segment_onset'].apply(
            lambda onset: floor(onset / 1000 * self.target_sr) / self.target_sr)
        segments['extended_offset'] = segments['segment_offset'].apply(
            lambda offset: ceil(offset / 1000 * self.target_sr) / self.target_sr)

        # Find better solution if more acoustic annotations are added in the future (concat dfs)
        pitch = pd.DataFrame.from_records(segments.apply(lambda row:
                                                   AcousticDerivator.get_pitch(
                                                       librosa.load(recording,
                                                                    mono=True,
                                                                    sr=self.target_sr,
                                                                    offset=row['extended_onset'],
                                                                    duration=(row['extended_offset'] - row['extended_onset']))[0],
                                                       self.target_sr,
                                                       func=AcousticDerivator.f2st
                                                   ), axis=1).tolist())

        # Drop raw pitch values
        pitch.drop(list(pitch.filter(regex='raw_')), axis=1, inplace=True)

        pitch.index = segments.index
        audio_segments = pd.concat([segments, pitch], axis=1)

        audio_segments.drop(columns=['extended_onset',
                                  'extended_offset'],
                         inplace=True)

        return audio_segments

    def get_auto_metadata(self,
                          am: 'AnnotationManager',
                          input_set: str,
                          output_set: str
                          ) -> dict:
        meta = {'segmentation': input_set,
                'has_acoustics': 'Y',
                'parameters': {'profile': self.profile, 'target_sr':self.target_sr},
                }

        if 'segmentation_type' in am.sets.columns:
            meta['segmentation_type'] = am.sets.loc[input_set,'segmentation_type']

        if 'has_speaker_type' in am.sets.columns:
            meta['has_speaker_type'] = am.sets.loc[input_set,'has_speaker_type']

        return meta


class ConversationDerivator(Derivator):
    """
    Derivator for predicting interactions between participants given an identified segmentation

    Based on the given interval (iti, maximum time elapsed after the end of an utterance for the next one to be
        considered an interaction) and delay (minimum time elapsed after the start of an utterance for the next
        one to be considered an interaction),
        classifies whether each segment is an interaction with the previous (columns is_CT i.e. is conversational turn).
         Then adds a column grouping vocalisations which belong to the same conversation (conv_count)
    """
    INTERACTIONS = {
        'CHI': {'FEM', 'MAL', 'OCH', 'CHI'},
        'FEM': {'FEM', 'MAL', 'OCH', 'CHI'},
        'MAL': {'FEM', 'MAL', 'OCH', 'CHI'},
        'OCH': {'FEM', 'MAL', 'OCH', 'CHI'},
    }

    def __init__(self,
                 interactions: dict = INTERACTIONS,
                 max_interval: int = 5000,
                 min_delay: int = 0,
                ):
        """
        :param interactions: dictionary mapping each speaker_type to the speaker_types it can interact with
        :type interactions: dict
        :param max_interval: maximum interval in ms for it to be considered a turn transition, default = 5000
        :type max_interval: int
        :param min_delay: minimum delay in ms from previous speaker start of vocalization from
         a vocalization to be considered a response to the previous one
        :type min_delay: int
        """
        self.interactions = interactions
        self.max_interval = max_interval
        self.min_delay = min_delay

        super().__init__()

    # Work in progress, method and parameters may evolve
    def derive(self,
               project: ChildProject,
               metadata: dict,
               segments: pd.DataFrame,
              ) -> pd.DataFrame:
        speakers = set(self.interactions.keys())

        segments = segments[segments["speaker_type"].isin(speakers)].copy()

        if segments.shape[0]:

            # store the duration between vocalizations
            segments["iti"] = segments["segment_onset"] - segments["segment_offset"].shift(1)
            # store the previous speaker
            segments["prev_speaker_type"] = segments["speaker_type"].shift(1)

            segments["delay"] = segments["segment_onset"] - segments["segment_onset"].shift(1)
            segments = segments.reset_index(drop=True)

            # each row is a turn transition if: 1) the speaker can interact with previous speaker, 2) it did not start
            # further than <max_interval> after the previous speaker stopped talking, 3) it did not begin earlier than
            # <delay> ms after the previous speaker started speaking
            # note that we allow iti to be negative, which means that a turn transition can exist when speaking before
            # the previous speaker finished talking
            segments["is_CT"] = (
                    (segments.apply(lambda row: row["prev_speaker_type"] in self.interactions[row['speaker_type']], axis=1))
                    &
                    (segments['iti'] < self.max_interval)
                    &
                    (segments['delay'] >= self.min_delay)
            )

            # find places where the sequence of turn transitions changes status to find beginning and ends of conversations
            diff = np.diff(segments['is_CT'].to_list() + [0])
            segments['diff'] = pd.Series(diff)
            segments['conv_number'] = segments['diff'][segments['diff'] == 1].cumsum().astype('Int64')
            segments['conv_count'] = segments[(segments['is_CT']) | (segments['diff'])][
                'conv_number'].interpolate(method='pad', limit_direction='forward')
            df = segments.drop(columns=['diff', 'conv_number'])

            return df
        else:
            return pd.DataFrame([], columns=['segment_onset', 'raw_filename', 'segment_offset'])

    def get_auto_metadata(self,
                          am: 'AnnotationManager',
                          input_set: str,
                          output_set: str
                          ) -> dict:
        meta= {'segmentation': input_set,
                'has_interactions': 'Y',
                'parameters': {'interactions': self.interactions,
                               'max_interval':self.max_interval,
                               'min_delay':self.min_delay},
                }

        if 'segmentation_type' in am.sets.columns:
            meta['segmentation_type'] = am.sets.loc[input_set,'segmentation_type']

        if 'has_speaker_type' in am.sets.columns:
            meta['has_speaker_type'] = am.sets.loc[input_set,'has_speaker_type']

        return meta

class RemoveOverlapsDerivator(Derivator):
    """
        Derivator going from permissive segmentation to restrictive, discarding any overlapping segments

        Cuts the segments to discard any part that has overlapping speech, resulting in a segmentation with no overlap
        of speech. Parts that contained overlapping speech therefore appear empty of any speech.
    """
    def __init__(self,
                speakers: List[str] = ['CHI', 'OCH', 'FEM', 'MAL'],
                ):
        """
        :param speakers: list of speakers to consider in speaker_type column,
        all the others will be completely ignored and removed (useful to remove
        <SPEECH> label for example)
        :type speakers: list[str]
        """
        self.speakers = speakers

        super().__init__()

    def derive(self,
               project: ChildProject,
               metadata: dict,
               segments: pd.DataFrame,
               ) -> pd.DataFrame:
        # restrict to wanted speakers  (remove SPEECH)
        segments = segments[segments['speaker_type'].isin(self.speakers)]
        segments = segments.sort_values(['segment_onset', 'segment_offset'])

        # initiate a new dataframe to concat into
        new_segments = pd.DataFrame(columns=segments.columns)

        for i, row in segments.iterrows():
            # print('new seg')
            # print(pd.DataFrame(row).transpose())

            # select all segments that overlap with the current vocalization
            overlapping_segments = segments[(segments['segment_onset'] < row['segment_offset']) &
                                            (segments['segment_offset'] > row['segment_onset']) &
                                            (segments.index != i)].sort_values('segment_onset')

            overlaps = []
            index = 0

            # squash overlapping into a single timeline:
            # we take the list of segments that are overlapping with the original segment
            # then we merge them to form a continuous 'times when overlaps exist' timeline
            while index < overlapping_segments.shape[0]:
                if index == 0 or (overlapping_segments.iloc[index]['segment_onset'] > overlaps[-1][1]):
                    overlaps.append((overlapping_segments.iloc[index]['segment_onset'],
                                     overlapping_segments.iloc[index]['segment_offset']))
                elif (overlapping_segments.iloc[index]['segment_onset'] <= overlaps[-1][1] and
                      overlapping_segments.iloc[index]['segment_offset'] > overlaps[-1][1]):
                    overlaps[-1] = (overlaps[-1][0], overlapping_segments.iloc[index]['segment_offset'])
                index += 1

            new_seg = pd.DataFrame(row.copy()).transpose()

            # Using the 'overlapping timeline' created previously, edit the original vocalization:
            # for every overlap segment, reconstruct the vocalization by removing overlapping parts
            for ovl in overlaps:
                # print('new ovl')
                # print(ovl)
                index = 0
                while index < new_seg.shape[0]:
                    # print(index)
                    if ovl[0] < new_seg.iloc[index]['segment_offset'] and ovl[1] > new_seg.iloc[index]['segment_onset']:
                        if ovl[0] <= new_seg.iloc[index]['segment_onset']:
                            if ovl[1] >= new_seg.iloc[index]['segment_offset']:
                                new_seg.drop(new_seg.iloc[index].name, inplace=True)
                                index -= 1
                            else:
                                new_seg.iloc[index]['segment_onset'] = ovl[1]
                        else:
                            mem_offset = new_seg.iloc[index]['segment_offset']
                            new_seg.iloc[index]['segment_offset'] = ovl[0]
                            if ovl[1] < mem_offset:
                                seg = pd.DataFrame(row.copy()).transpose()
                                seg['segment_onset'] = ovl[1]
                                seg['segment_offset'] = mem_offset
                                new_seg = pd.concat([new_seg.iloc[:index + 1],
                                                     seg,
                                                     new_seg.iloc[index + 1:]]
                                                    ).reset_index(drop=True)
                    # else:
                    # print('skip {} {}'.format(new_seg.iloc[index]['segment_onset'],new_seg.iloc[index]['segment_offset']))
                    index += 1

            new_segments = pd.concat([new_segments, new_seg]).reset_index(drop=True)

        return new_segments

    def get_auto_metadata(self,
                          am: 'AnnotationManager',
                          input_set: str,
                          output_set: str
                          ) -> dict:
        meta= {'segmentation': output_set,
                'segmentation_type' : 'restrictive',
                'parameters': {'speakers': self.speakers},
                }

        if 'has_speaker_type' in am.sets.columns:
            meta['has_speaker_type'] = am.sets.loc[input_set,'has_speaker_type']

        return meta

class CVADerivator(Derivator):
    """
        Derivator predicting addressed character of identified vocalizations

        takes a dataframe of annotation segments as an input and based on the given iti (inter turn
        interval) and scenario (permissive or restrictive),
        classifies whether each annotation is targeted to the key child or overheard. Filling in the column cva (child
        vocalization adjacent), Y meaning it is in an interaction with the child, N meaning the vocalization is not in
        direct interaction with the key child.
    """
    def __init__(self,
               iti: int =5000,
               scenario: str ='R',
                 ):
        """
        :param iti: maximum interval in ms for it to be considered interaction, default = 5000
        :type iti: int
        :param scenario: scenario to choose from P for permissive, R for restrictive. You MUST use annotations that are
        respectively permissive (allow overlaps between speakers) and restrictive (no overlap allowed between speakers) for
        those scenarios
        :type scenario: str
        """
        self.iti = iti
        self.scenario = scenario

        super().__init__()

    def derive(self,
               project: ChildProject,
               metadata: dict,
               segments: pd.DataFrame
               ) -> pd.DataFrame:

        def classify_speaker_type(speaker_type):
            return 'C' if pd.isna(speaker_type) or speaker_type == 'CHI' else 'O'

        segments['speaker_class'] = segments['speaker_type'].apply(classify_speaker_type)
        segments['cva'] = 'N'
        debug_data = []

        for idx in range(len(segments)):
            curr_row = segments.iloc[idx]
            prev_row = segments.iloc[idx - 1] if idx > 0 else None
            next_row = segments.iloc[idx + 1] if idx < len(segments) - 1 else None

            prev_gap = curr_row['segment_onset'] - prev_row['segment_offset'] if prev_row is not None else None
            next_gap = next_row['segment_onset'] - curr_row['segment_offset'] if next_row is not None else None

            # R scenario
            if self.scenario == "R":
                if curr_row["speaker_class"] == "C":
                    segments.at[idx, 'cva'] = 'NA'
                elif prev_row is not None and prev_row['speaker_class'] == 'C' and prev_gap is not None and prev_gap <= self.iti:
                    segments.at[idx, 'cva'] = 'Y'

                elif next_row is not None and next_row['speaker_class'] == 'C' and next_gap is not None and next_gap <= self.iti:
                    segments.at[idx, 'cva'] = 'Y'
                else:
                    segments.at[idx, 'cva'] = 'N'

            # P scenario
            else:
                if curr_row["speaker_class"] == "C":
                    # If the current speaker is a child, classify as CHI
                    segments.at[idx, 'cva'] = 'NA'
                else:
                    # Check if there is a child (CHI) before or after the current speaker
                    has_child_neighbor = (
                            (prev_row is not None and prev_row['speaker_class'] == "C") or
                            (next_row is not None and next_row['speaker_class'] == "C")
                    )

                    if has_child_neighbor:
                        # If there is a child before or after, automatically classify as Y
                        segments.at[idx, 'cva'] = 'Y'
                    else:
                        # Otherwise, apply standard rules
                        prev_is_turn = prev_row is not None and abs(prev_gap) <= self.iti
                        next_is_turn = next_row is not None and abs(next_gap) <= self.iti

                        if prev_is_turn and next_is_turn:
                            # Choose the smallest gap in absolute value
                            if abs(prev_gap) < abs(next_gap):
                                # Use previous row for cva
                                if prev_row['speaker_class'] != "C" and curr_row["speaker_type"] != prev_row[
                                    'speaker_type']:
                                    segments.at[idx, 'cva'] = 'N'
                                else:
                                    segments.at[idx, 'cva'] = 'Y'
                            else:
                                # Use next row for classification of cvs
                                if next_row['speaker_class'] != "C" and curr_row["speaker_type"] != next_row[
                                    'speaker_type']:
                                    segments.at[idx, 'cva'] = 'N'
                                else:
                                    segments.at[idx, 'cva'] = 'Y'
                        elif prev_is_turn:
                            if prev_row['speaker_class'] != "C" and curr_row["speaker_type"] != prev_row['speaker_type']:
                                segments.at[idx, 'cva'] = 'N'
                            else:
                                segments.at[idx, 'cva'] = 'Y'
                        elif next_is_turn:
                            if next_row['speaker_class'] != "C" and curr_row["speaker_type"] != next_row['speaker_type']:
                                segments.at[idx, 'cva'] = 'N'
                            else:
                                segments.at[idx, 'cva'] = 'Y'
                        else:
                            # Default to Y
                            segments.at[idx, 'cva'] = 'Y'

        return segments

    def get_auto_metadata(self,
                          am: 'AnnotationManager',
                          input_set: str,
                          output_set: str
                          ) -> dict:
        meta = {'segmentation': output_set,
                'has_addressee' : 'Y',
                'parameters': {'iti': self.iti, 'scenario': self.scenario}
                }

        if 'segmentation_type' in am.sets.columns:
            meta['segmentation_type'] = am.sets.loc[input_set,'segmentation_type']

        if 'has_speaker_type' in am.sets.columns:
            meta['has_speaker_type'] = am.sets.loc[input_set,'has_speaker_type']
    
        return meta


# listing the possible derivators available by default
DERIVATIONS = {
    "acoustics": AcousticDerivator,
    "conversations": ConversationDerivator,
    "remove-overlaps": RemoveOverlapsDerivator,
    "cva": CVADerivator,
}
