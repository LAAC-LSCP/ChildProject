import numpy as np
import pandas as pd


def acoustics():
    pass

def conversations(annotations: pd.DataFrame,
        interlocutors_1=('CHI',),
        interlocutors_2=('FEM', 'MAL', 'OCH'),
        max_interval=1000,
        min_delay=0):

    """

    :param annotations: dataframe of annotations
    :type annotations: DataFrame
    :param interlocutors_1: first group of interlocutors, default = ('CHI')
    :type interlocutors_1: tuple
    :param interlocutors_2: second group of interlocutors, default = ('FEM','MAL','OCH')
    :type interlocutors_2: tuple
    :param max_interval: maximum interval in ms for it to be considered a turn, default = 1000
    :type max_interval: int
    :param min_delay: minimum delay between somebody starting speaking
    :type min_delay: int

    :return: output annotation dataframe
    :rtype: DataFrame
    """

    speakers = set(interlocutors_1 + interlocutors_2)

    interactants = {k: set(interlocutors_2) for k in interlocutors_1}

    for k in interlocutors_2:
        if k in interactants:
            interactants[k] = interactants[k] | set(interlocutors_1)
        else:
            interactants[k] = set(interlocutors_1)

    annotations = annotations[annotations["speaker_type"].isin(speakers)].copy()

    if annotations.shape[0]:

        # store the duration between vocalizations
        annotations["iti"] = annotations["segment_onset"] - annotations["segment_offset"].shift(1)
        # store the previous speaker
        annotations["prev_speaker_type"] = annotations["speaker_type"].shift(1)

        annotations["delay"] = annotations["segment_onset"] - annotations["segment_onset"].shift(1)
        annotations = annotations.reset_index(drop=True)
        annotations['idx'] = annotations.index

        # not using absolute value for 'iti' is a choice and should be evaluated (we allow speakers to 'interrupt'
        # themselves
        annotations["is_CT"] = (
                (annotations.apply(lambda row: row["prev_speaker_type"] in interactants[row['speaker_type']], axis=1))
                &
                (annotations['iti'] < max_interval)
                &
                (annotations['delay'] >= min_delay)
        )

        diff = np.diff(annotations['is_CT'].to_list() + [0])
        annotations['diff'] = pd.Series(diff)
        annotations['conv_number'] = annotations['diff'][annotations['diff'] == 1].cumsum().astype('Int64')
        annotations['conv_count'] = annotations[(annotations['is_CT']) | (annotations['diff'])][
            'conv_number'].interpolate(method='pad', limit_direction='forward')
        annotations = annotations[['segment_onset', 'segment_offset', 'raw_filename', 'iti', 'delay', 'is_CT', 'conv_count']]
        print(annotations)
        return annotations
    else:
        return pd.DataFrame([])

DERIVATIONS = {
    "acoustics": acoustics,
    "conversations": conversations,
}
