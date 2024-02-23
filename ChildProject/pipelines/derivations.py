import numpy as np
import pandas as pd


def acoustics():
    pass

def conversations(annotations: pd.DataFrame,
        interlocutors_1=('CHI',),
        interlocutors_2=('FEM', 'MAL', 'OCH'),
        max_interval=1000,
        min_delay=0):

    """import and convert ``annotation``. This function should not be called outside of this class.

    :param import_function: If callable, ``import_function`` will be called to convert the input annotation into a dataframe. Otherwise, the conversion will be performed by a built-in function.
    :type import_function: Callable[[str], pd.DataFrame]
    :param output_set: name of the new set of derived annotations
    :type output_set: str
    :param params: Optional parameters. With ```new_tiers```, the corresponding EAF tiers will be imported
    :type params: dict
    :param annotation: input annotation dictionary (attributes defined according to :ref:`ChildProject.annotations.AnnotationManager.SEGMENTS_COLUMNS`)
    :type annotation: dict
    :param overwrite_existing: choose if lines with the same set and annotation_filename should be overwritten
    :type overwrite_existing: bool
    :return: output annotation dictionary (attributes defined according to :ref:`ChildProject.annotations.AnnotationManager.SEGMENTS_COLUMNS`)
    :rtype: dict
    """

    """number of conversational turn counts based on vocalizations occurring
    in a given interval of one another

    keyword arguments:
        - interlocutors_1 : first group of interlocutors, default = ['CHI']
        - interlocutors_2 : second group of interlocutors, default = ['FEM','MAL','OCH']
        - max_interval : maximum interval in ms for it to be considered a turn, default = 1000
        - min_delay : minimum delay between somebody starting speaking
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
        annotations['inter_conv'] = annotations[(annotations['is_CT']) | (annotations['diff'])][
            'conv_number'].interpolate(method='pad', limit_direction='forward')

        return annotations
    else:
        return pd.DataFrame([])

DERIVATIONS = {
    "acoustics": acoustics,
    "conversations": conversations,
}
