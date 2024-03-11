# define functions to calculate metrics
import pandas as pd
import numpy as np
import ast
import re
import functools
from typing import Union, Set, Tuple

"""
This file lists all the metrics functions commonly used.
New metrics can be added by defining new functions for the Conversations class to use :
 - Create a new function using the same arguments (i.e. annotations, duration, **kwargs)
 - Define calculation of the metric with:
     - annotations, which is a dataframe containing all the relevant annotated segments  to use. It contains the
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
#TODO
# 1. Start and end time of each conversation in the recording
# 2. Duration of time between conversations (e.g., time between Convo 1 and Convo 2)
# 3. Key Child ID (i.e., some identifier for the key child in the data set)
# 4. Recording ID (i.e., which of the Key Child's recordings is this, if the Key Child has multiple recordings)
# 5. A string with a list of speaker tags in the conversation (e.g., "CHI, FEM, OCH")

def conversationFunction(args: set = set()):
    """Decorator for all metrics functions to make them ready to be called by the pipeline.

    :param args: set of required keyword arguments for that function, raise ValueError if were not given \
    you cannot use keywords [name, callable, set] as they are reserved
    :type args: set
    :return: new function to substitute the metric function
    :rtype: Callable
    """

    def decorator(function):
        for a in args:
            if a in RESERVED:
                raise ValueError(
                    'Error when defining {} with required argument {}, you cannot use reserved keywords {},\
                     change your required argument name'.format(
                        function.__name__, a, RESERVED))

        @functools.wraps(function)
        def new_func(annotations: pd.DataFrame, **kwargs):
            for arg in args:
                if arg not in kwargs:
                    raise ValueError(f"{function.__name__} metric needs an argument <{arg}>")

            res = function(annotations, **kwargs)

            return res

        return new_func

    return decorator


@conversationFunction()
def conversation_onset(annotations: pd.DataFrame):
    return annotations.reset_index().iloc[0]['segment_onset']


@conversationFunction()
def conversation_offset(annotations: pd.DataFrame):
    return annotations.reset_index().iloc[-1]['segment_offset']


@conversationFunction()
def conversation_duration(annotations: pd.DataFrame):
    return annotations.reset_index().iloc[-1]['segment_offset'] - annotations.reset_index().iloc[0]['segment_onset']


@conversationFunction()
def vocalisations_count(annotations: pd.DataFrame):
    return annotations['speaker_type'].count()


@conversationFunction()
def who_initiated(annotations: pd.DataFrame):
    return annotations.reset_index().iloc[0]['speaker_type']


@conversationFunction()
def who_finished(annotations: pd.DataFrame):
    return annotations.reset_index().iloc[-1]['speaker_type']


@conversationFunction()
def total_duration_of_vocalisations(annotations: pd.DataFrame):
    return annotations['voc_duration'].sum()


@conversationFunction({'speaker'})
def is_speaker(annotations: pd.DataFrame, **kwargs):
    return kwargs["speaker"] in annotations['speaker_type'].tolist()


@conversationFunction({'speaker'})
def voc_counter(annotations: pd.DataFrame, **kwargs):
    return annotations[annotations['speaker_type'] == kwargs["speaker"]]['speaker_type'].count()


@conversationFunction({'speaker'})
def voc_total(annotations: pd.DataFrame, **kwargs):
    return annotations[annotations['speaker_type'] == kwargs["speaker"]]['voc_duration'].sum(min_count=1)


@conversationFunction({'speaker'})
def voc_contribution(annotations: pd.DataFrame, **kwargs):
    speaker_total = annotations[annotations['speaker_type'] == kwargs["speaker"]]['voc_duration'].sum(min_count=1)
    total = annotations['voc_duration'].sum()
    return speaker_total / total


@conversationFunction()
def assign_conv_type(annotations: pd.DataFrame):
    #pd.Categorical(['overheard', 'dyadic_FEM', 'dyadic_MAL', 'peer', 'parent', 'triadic_FEM', 'triadic_MAL', 'multiparty'])
    speaker_present = {}
    for speaker in ['CHI', 'FEM', 'MAL', 'OCH']:
        speaker_present[speaker] = [speaker in annotations['speaker_type'].tolist()]
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
    return np.nan()



