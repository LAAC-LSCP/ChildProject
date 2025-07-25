import pandas as pd
import numpy as np
import ast
import re
import functools
from typing import Union, Set, Tuple

"""
This file lists all the metrics functions commonly used.
New metrics can be added by defining new functions for the Conversations class to use :
 - Create a new function using the same arguments (i.e. segments, duration, **kwargs)
 - Define calculation of the metric with:
     - segments, which is a dataframe containing all the relevant annotated segments  to use. It contains the
       annotation content (https://childproject.readthedocs.io/en/latest/format.html#id10) joined with the annotation
       index info (https://childproject.readthedocs.io/en/latest/format.html#id11) as well as any column that was
       requested to be added to the results by the user using --child-cols or --rec-cols (eg --child-cols child_dob,
       languages will make columns 'child_dob' and 'languages' available)
     - duration which is the duration of audio annotated in milliseconds
     - kwargs, whatever keyword parameter you chose to pass to the function (except 'name', 'callable', 'set' which can 
       not be used). This will need to be given with the list of metrics when called
 - Wrap you function with the 'conversationFunction' decorator to make it callable by the pipeline, read conversationFunction help 
   for more info

!! Metrics functions should still behave and return the correct result when receiving an empty dataframe
"""
RESERVED = {'name', 'callable'}  # arguments reserved usage. use other keyword labels.


def conversationFunction(args: set = set()) -> callable:
    """Decorator for all metrics functions to make them ready to be called by the pipeline.

    :param args: set of required keyword arguments for that function, raise ValueError if were not given \
    you cannot use keywords [name, callable, set] as they are reserved
    :type args: set
    :return: new function to substitute the metric function
    :rtype: Callable
    """

    def decorator(function) -> callable:
        for a in args:
            if a in RESERVED:
                raise ValueError(
                    'Error when defining {} with required argument {}, you cannot use reserved keywords {},\
                     change your required argument name'.format(
                        function.__name__, a, RESERVED))

        @functools.wraps(function)
        def new_func(segments: pd.DataFrame, **kwargs):
            for arg in args:
                if arg not in kwargs:
                    raise ValueError(f"{function.__name__} metric needs an argument <{arg}>")

            res = function(segments, **kwargs)

            return res

        return new_func

    return decorator


@conversationFunction()
def who_initiated(segments: pd.DataFrame) -> str:
    """speaker type who spoke first in the conversation

    Required keyword arguments:
    """
    return segments.iloc[0]['speaker_type']


@conversationFunction()
def who_finished(segments: pd.DataFrame) -> str:
    """speaker type who spoke last in the conversation

    Required keyword arguments:
    """
    return segments[segments['segment_offset'] == segments['segment_offset'].max()].iloc[0]['speaker_type']

@conversationFunction()
def participants(segments: pd.DataFrame) -> str:
    """list of speakers participating in the conversation, '/' separated

    Required keyword arguments:
    """
    return '/'.join(segments['speaker_type'].unique())

@conversationFunction()
def voc_total_dur(segments: pd.DataFrame) -> int:
    """summed duration of all speech in the conversation (ms) N.B. can be higher than conversation duration as
    speakers may speak at the same time, resulting in multiple spoken segments happening simultaneously

    Required keyword arguments:
    """
    return segments['voc_duration'].sum()


@conversationFunction({'speaker'})
def is_speaker(segments: pd.DataFrame, **kwargs) -> bool:
    """is a specific speaker type present in the conversation

    Required keyword arguments:
        - speaker : speaker_type label
    """
    return kwargs["speaker"] in segments['speaker_type'].tolist()


@conversationFunction({'speaker'})
def voc_speaker_count(segments: pd.DataFrame, **kwargs) -> int:
    """number of vocalizations produced by a given speaker

    Required keyword arguments:
        - speaker : speaker_type label
    """
    return segments[segments['speaker_type'] == kwargs["speaker"]]['speaker_type'].count()


@conversationFunction({'speaker'})
def voc_speaker_dur(segments: pd.DataFrame, **kwargs) -> int:
    """summed duration of speech for a given speaker in the conversation

    Required keyword arguments:
        - speaker : speaker_type label
    """
    return segments[segments['speaker_type'] == kwargs["speaker"]]['voc_duration'].sum(min_count=1)


@conversationFunction({'speaker'})
def voc_dur_contribution(segments: pd.DataFrame, **kwargs) -> float:
    """contribution of a given speaker in the conversation compared to others, in terms of total speech duration

    Required keyword arguments:
        - speaker : speaker_type label
    """
    speaker_total = segments[segments['speaker_type'] == kwargs["speaker"]]['voc_duration'].sum(min_count=1)
    total = segments['voc_duration'].sum()
    return speaker_total / total


@conversationFunction()
def assign_conv_type(segments: pd.DataFrame) -> str:
    """Compute the conversation type (overheard, dyadic_XXX, peer, parent, triadic_XXX, multiparty) depending on the
    participants

    Required keyword arguments:
    """
    #pd.Categorical(['overheard', 'dyadic_FEM', 'dyadic_MAL', 'peer', 'parent', 'triadic_FEM', 'triadic_MAL', 'multiparty'])
    speaker_present = {}
    for speaker in ['CHI', 'FEM', 'MAL', 'OCH']:
        speaker_present[speaker] = [speaker in segments['speaker_type'].tolist()]
    speaker_df = pd.DataFrame.from_dict(speaker_present).iloc[0, :]

    if not speaker_df['CHI']:
        return 'overheard'

    elif speaker_df['CHI']:
        if not speaker_df['OCH'] and speaker_df[['FEM', 'MAL']].sum() == 1:
            if speaker_df['FEM']:
                return 'dyadic_FEM'

            if speaker_df['MAL']:
                return 'dyadic_MAL'

        if speaker_df['OCH'] and speaker_df[['FEM', 'MAL']].sum() == 0:
            return 'peer'

        if not speaker_df['OCH'] and speaker_df[['FEM', 'MAL']].sum() == 2:
            return 'parent'

        if speaker_df['OCH'] and speaker_df[['FEM', 'MAL']].sum() == 1:
            if speaker_df['FEM']:
                return 'triadic_FEM'
            if speaker_df['MAL']:
                return 'triadic_MAL'

        if speaker_df[['OCH', 'FEM', 'MAL']].sum() == 3:
            return 'multiparty'
    return np.nan



