from math import ceil, floor

import numpy as np
import pandas as pd
import librosa

from ChildProject.projects import STANDARD_PROFILE, STANDARD_SAMPLE_RATE


@np.vectorize
def f2st(f_Hz, base=50):
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
    f2st.__name__ = 'semitone'

    semi = np.log(2 ** (1 / 12))
    return (np.log(f_Hz) - np.log(base)) / semi

def get_pitch(audio_time_series, sampling_rate, func=None):
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

def acoustics(project,
              metadata: dict,
              annotations: pd.DataFrame,
              profile=STANDARD_PROFILE,
              target_sr=STANDARD_SAMPLE_RATE,
              ):
    """
        Read an audio file and returns the audio time series and its sampling rate
        :param file_path: path to an audio file
        :type file_path: str
        :return: (audio time series, sampling rate)
        :rtype: np.array
    """
    recording = project.get_recording_path(metadata['recording_filename'], profile=profile)
    file_sr = librosa.get_samplerate(recording)
    assert file_sr == target_sr, ValueError("Mismatch between file's true sampling rate ({}) and "
                                            "target sampling rate ({})!".format(file_sr, target_sr))
    audio_time_series, sampling_rate = librosa.load(recording, mono=True, sr=target_sr)

    # Computes the start frame and end frame of the given segments given is on/offset in seconds
    annotations['frame_onset'] = annotations['segment_onset'].apply(
        lambda onset: floor(onset / 1000 * sampling_rate))
    annotations['frame_offset'] = annotations['segment_offset'].apply(
        lambda offset: ceil(offset / 1000 * sampling_rate))

    # Find better solution if more acoustic annotations are added in the future (concat dfs)
    pitch = pd.DataFrame.from_records(annotations.apply(lambda row:
                                               get_pitch(
                                                   audio_time_series[row['frame_onset']:row['frame_offset']],
                                                   target_sr,
                                                   func=f2st
                                               ), axis=1).tolist())

    # Drop raw pitch values
    pitch.drop(list(pitch.filter(regex='raw_')), axis=1, inplace=True)

    pitch.index = annotations.index
    audio_segments = pd.concat([annotations, pitch], axis=1)

    audio_segments.drop(columns=['frame_onset',
                              'frame_offset'],
                     inplace=True)

    return audio_segments



INTERACTIONS = {
    'CHI': {'FEM', 'MAL', 'OCH', 'CHI'},
    'FEM': {'FEM', 'MAL', 'OCH', 'CHI'},
    'MAL': {'FEM', 'MAL', 'OCH', 'CHI'},
    'OCH': {'FEM', 'MAL', 'OCH', 'CHI'},
}
# Work in progress, method and parameters may evolve
def conversations(project,
                  metadata: dict,
                  annotations: pd.DataFrame,
                  interactions=INTERACTIONS,
                  max_interval=5000,
                  min_delay=0):

    """ The function takes a dataframe of annotations as an input and based on the given interval and delay, classifies
    whether each annotation is a part of the conversation. Then adds a column grouping vocalisations which belong to
    the same conversation

    :param metadata: series mapping all the metadata available
    :type metadata: pd.Series
    :param annotations: dataframe of annotations
    :type annotations: DataFrame
    :param interactions: dictionary mapping each speaker_type to the speaker_types it can interact with
    :type interactions: dict
    :param max_interval: maximum interval in ms for it to be considered a turn transition, default = 5000
    :type max_interval: int
    :param min_delay: minimum delay in ms from previous speaker start of vocalization from
     a vocalization to be considered a response to the previous one
    :type min_delay: int

    :return: output annotation DataFrame
    :rtype: DataFrame
    """
    speakers = set(interactions.keys())

    annotations = annotations[annotations["speaker_type"].isin(speakers)].copy()

    if annotations.shape[0]:

        # store the duration between vocalizations
        annotations["iti"] = annotations["segment_onset"] - annotations["segment_offset"].shift(1)
        # store the previous speaker
        annotations["prev_speaker_type"] = annotations["speaker_type"].shift(1)

        annotations["delay"] = annotations["segment_onset"] - annotations["segment_onset"].shift(1)
        annotations = annotations.reset_index(drop=True)

        # each row is a turn transition if: 1) the speaker can interact with previous speaker, 2) it did not start
        # further than <max_interval> after the previous speaker stopped talking, 3) it did not begin earlier than
        # <delay> ms after the previous speaker started speaking
        # note that we allow iti to be negative, which means that a turn transition can exist when speaking before
        # the previous speaker finished talking
        annotations["is_CT"] = (
                (annotations.apply(lambda row: row["prev_speaker_type"] in interactions[row['speaker_type']], axis=1))
                &
                (annotations['iti'] < max_interval)
                &
                (annotations['delay'] >= min_delay)
        )

        # find places where the sequence of turn transitions changes status to find beginning and ends of conversations
        diff = np.diff(annotations['is_CT'].to_list() + [0])
        annotations['diff'] = pd.Series(diff)
        annotations['conv_number'] = annotations['diff'][annotations['diff'] == 1].cumsum().astype('Int64')
        annotations['conv_count'] = annotations[(annotations['is_CT']) | (annotations['diff'])][
            'conv_number'].interpolate(method='pad', limit_direction='forward')
        df = annotations.drop(columns=['diff', 'conv_number'])

        return df
    else:
        return pd.DataFrame([])


def remove_overlaps(project,
                    metadata: dict,
                    annotations,
                    speakers=['CHI', 'OCH', 'FEM', 'MAL'],
                    ):
    """takes a pandas dataframe of annotations containing at least the columns
    speaker_type segment_onset and segment_offset.
    Cuts the vocalizations to discard any part that has overlapping speech
    return the new dataframe of annotations

    :param df: Dataframe of annotations with speaker_type, segment_onset and
    segment_offset
    :type df: pd.DataFrame
    :param speakers: list of speakers to consider in speaker_type column,
    all the others will be completely ignored and removed (useful to remove
    <SPEECH> label for example)
    :type speakers: list[str]
    """
    # restrict to wanted speakers  (remove SPEECH)
    annotations = annotations[annotations['speaker_type'].isin(speakers)]
    segments = annotations.sort_values(['segment_onset', 'segment_offset'])

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
        # then we merge them to form a continuous 'times where overlaps exist' timeline
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


DERIVATIONS = {
    "acoustics": acoustics,
    "conversations": conversations,
    "remove-overlaps": remove_overlaps,
}
