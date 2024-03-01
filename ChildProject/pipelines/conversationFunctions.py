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

# error message in case of missing columns in annotations
MISSING_COLUMNS = 'The given set <{}> does not have the required column(s) <{}> for computing the {} metric'

RESERVED = {'set', 'name', 'callable'}  # arguments reserved usage. use other keyword labels.


def conversationFunction(args: set, columns: Union[Set[str], Tuple[Set[str], ...]], empty_value=0, default_name: str = None):
    """Decorator for all metrics functions to make them ready to be called by the pipeline.

    :param args: set of required keyword arguments for that function, raise ValueError if were not given \
    you cannot use keywords [name, callable, set] as they are reserved
    :type args: set
    :param columns: required columns in the dataframe given, missing columns raise ValueError
    :type columns: set
    :param default_name: default name to use for the metric in the resulting dataframe. Every keyword argument found in the name will be replaced by its value (e.g. 'voc_speaker_ph' uses kwarg 'speaker' so if speaker = 'CHI', name will be 'voc_chi_ph'). if no name is given, the __name__ of the function is used
    :type default_name: str
    :param empty_value: value to return when annotations are empty but the unit was annotated (e.g. 0 for counts like voc_speaker_ph , None for proportions like lp_n)
    :type empty_value: float|int
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
        def new_func(annotations: pd.DataFrame, duration: int, **kwargs):
            for arg in args:
                if arg not in kwargs:
                    raise ValueError(f"{function.__name__} metric needs an argument <{arg}>")
            # if a name is explicitly given, use it
            if 'name' in kwargs and not pd.isnull(kwargs['name']) and kwargs['name']:
                metric_name = kwargs['name']
            # else if a default name for the function exists, use the function name
            elif default_name:
                metric_name = default_name
            # else, no name was found, use the name of the function
            else:
                metric_name = function.__name__

            metric_name_replaced = metric_name
            # metric_name is the basename used to designate this metric (voc_speaker_ph),
            # metric_name_replaced replaces the values of kwargs
            # found in the name by their values, giving the metric name for that instance only (voc_chi_ph)
            for arg in kwargs:
                metric_name_replaced = re.sub(arg, str(kwargs[arg]).lower(), metric_name_replaced)
            if annotations.shape[0]:
                # if multiple possibilities of columns, explore each and fail only if each combination is missing
                # a column, if one possibility, fail if a column is missing
                if isinstance(columns, tuple) and len(columns) > 0 and isinstance(columns[0], set):
                    missing_columns = []
                    for possible_cols in columns:
                        possible_missing = possible_cols - set(annotations.columns)
                        if possible_missing:
                            missing_columns.append(possible_missing)
                    # if we have as many cases of missing columns as possibilities, we can't compute the metric
                    if len(missing_columns) == len(columns):
                        raise ValueError(
                            MISSING_COLUMNS.format(annotations['set'].iloc[0],
                                                   ' or '.join([str(s) for s in missing_columns]),
                                                   metric_name))
                else:
                    missing_columns = columns - set(annotations.columns)
                    if missing_columns:
                        raise ValueError(
                            MISSING_COLUMNS.format(annotations['set'].iloc[0], missing_columns, metric_name))
                res = function(annotations, duration, **kwargs)
            else:  # no annotation for that unit
                res = empty_value if duration else None  # duration != 0 => was annotated but not segments there
            return metric_name_replaced, res

        return new_func

    return decorator

@conversationFunction(set(), {"speaker_type", "conv_count", "duration"}, np.nan)
def is_speaker(annotations: pd.DataFrame, **kwargs):
    return kwargs["speaker"] in annotations['speaker_type'].tolist()

@conversationFunction(set(), {"speaker_type", "conv_count", "duration"}, np.nan)
def voc_counter(annotations: pd.DataFrame, **kwargs):
    return annotations[annotations['speaker_type'] == kwargs["speaker"]]['speaker_type'].count()

@conversationFunction(set(), {"speaker_type", "conv_count", "duration"}, np.nan)
def voc_total(annotations: pd.DataFrame, **kwargs):
    return annotations[annotations['speaker_type'] == kwargs["speaker"]]['voc_duration'].sum(min_count=1)

@conversationFunction(set(), {"speaker_type", "conv_count", "duration"}, np.nan)
def assign_conv_type(conv):
    if not conv['CHI_present']:
        return 'overheard'
    elif conv['CHI_present']:
        if not conv['OCH_present'] and conv[['FEM_present', 'MAL_present']].sum() == 1:
            if conv['FEM_present']:
                return 'dyadic_FEM'
            if conv['MAL_present']:
                return 'dyadic_MAL'
        if conv['OCH_present'] and conv[['FEM_present', 'MAL_present']].sum() == 0:
            return 'peer'
        if not conv['OCH_present'] and conv[['FEM_present', 'MAL_present']].sum() == 2:
            return 'parent'
        if conv['OCH_present'] and conv[['FEM_present', 'MAL_present']].sum() == 1:
            if conv['FEM_present']:
                return 'triadic_FEM'
            if conv['MAL_present']:
                return 'triadic_MAL'
        if conv[['OCH_present', 'FEM_present', 'MAL_present']].sum() == 3:
            return 'multiparty'
    return np.nan()
# @conversationFunction(set(), {"speaker_type", "conv_count", "duration"}, np.nan)
# def voc_average(annotations: pd.DataFrame, **kwargs):
#     return annotations[annotations['speaker_type'] == kwargs["speaker"]]['voc_duration'].mean()


