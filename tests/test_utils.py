#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 11:23:05 2022

@author: lpeurey
"""
import pytest

from ChildProject.projects import ChildProject
from ChildProject.utils import series_to_datetime, time_intervals_intersect, intersect_ranges, TimeInterval, get_audio_duration, calculate_shift, find_lines_involved_in_overlap
import pandas as pd
import datetime
import os

def test_series_to_datetime():
    project = ChildProject("examples/valid_raw_data")
    
    only_time = pd.Series(['3:12','04:14', '12:19:21', '32:77'])
    only_date = pd.Series(['2022-01-23','2022-01-23', '2022-01-23', '2022-01-23'])
    
    truth = pd.Series([datetime.datetime(1900,1,1,3,12,0,0),
                       datetime.datetime(1900,1,1,4,14,0,0),
                       datetime.datetime(1900,1,1,12,19,21,0),
                       pd.NaT,
                       ])
    
    #convert to datetime using the formats for start_time listed in the project RECORDINGS_COLUMNS.Index(name == 'start_time')
    converted_time = series_to_datetime(only_time, project.RECORDINGS_COLUMNS, 'start_time')
    
    pd.testing.assert_series_equal(converted_time, truth, check_index_type=False, check_dtype=False)
    
    truth = pd.Series([datetime.datetime(2022,1,23,3,12,0,0),
                       datetime.datetime(2022,1,23,4,14,0,0),
                       datetime.datetime(2022,1,23,12,19,21,0),
                       pd.NaT,
                       ])
    
    #convert to datetime using the formats for start_time and date_iso listed in the project RECORDINGS_COLUMNS.Index(name == 'start_time') and Index(name == 'date_iso')
    converted_time = series_to_datetime(only_time, project.RECORDINGS_COLUMNS, 'start_time', only_date, project.RECORDINGS_COLUMNS, 'date_iso')
    
    pd.testing.assert_series_equal(converted_time, truth, check_index_type=False, check_dtype=False)
    
def test_time_intervals_intersect():
    truth = [TimeInterval(datetime.datetime(1900,1,1,10,36),datetime.datetime(1900,1,1,21,4))]
    res = time_intervals_intersect(TimeInterval(datetime.datetime(1900,1,1,8,57),datetime.datetime(1900,1,1,21,4)),TimeInterval(datetime.datetime(1900,1,1,10,36),datetime.datetime(1900,1,1,22,1)))
    
    for i in range(len(truth)):
        assert truth[i] == res[i] 
    
    truth = [TimeInterval(datetime.datetime(1900,1,1,8,57),datetime.datetime(1900,1,1,10,36)),TimeInterval(datetime.datetime(1900,1,1,21,4),datetime.datetime(1900,1,1,22,1))]
    res = time_intervals_intersect(TimeInterval(datetime.datetime(1900,1,1,8,57),datetime.datetime(1900,1,1,22,1)),TimeInterval(datetime.datetime(1900,1,1,21,4),datetime.datetime(1900,1,1,10,36)))
    
    for i in range(len(truth)):
        assert truth[i] == res[i]
    
    truth = [TimeInterval(datetime.datetime(1900,1,1,21,4),datetime.datetime(1900,1,1,3,1))]
    res = time_intervals_intersect(TimeInterval(datetime.datetime(1900,1,1,8,57),datetime.datetime(1900,1,1,3,1)),TimeInterval(datetime.datetime(1900,1,1,21,4),datetime.datetime(1900,1,1,7,36)))
    
    for i in range(len(truth)):
        assert truth[i] == res[i]

AUDIOS = os.path.join('examples','valid_raw_data','recordings','raw')
@pytest.mark.parametrize('file,result',
    [(os.path.join(AUDIOS,'sound.wav'), 4),
     (os.path.join(AUDIOS,'unindexed.mp3'), 0),
     (os.path.join(AUDIOS,'non_existing.wav'), 0),
    ]) 
def test_get_audio_duration(file, result):
    
    assert result == get_audio_duration(file)

AUDIOS_VALID = os.path.join('examples','valid_raw_data','recordings','raw')
AUDIOS_INVALID = os.path.join('examples','invalid_raw_data','recordings','raw')
@pytest.mark.parametrize('f1,f2,result',
    [(os.path.join(AUDIOS_VALID,'sound.wav'),os.path.join(AUDIOS_VALID,'sound2.wav'),(169.9779200946652, 1639 )),
     (os.path.join(AUDIOS_VALID,'sound.wav'),os.path.join(AUDIOS_INVALID,'test_1_20160918.wav'),(5.9604644775390625e-05, 1024 )),
     (os.path.join(AUDIOS_VALID,'sound2.wav'),os.path.join(AUDIOS_INVALID,'test_1_20160918.wav'),(163.95631432533264, 512)
),
    ])
def test_calculate_shift(f1,f2, result):
    
    assert result == calculate_shift(f1,f2,0,0,4)
    
INP_CSV = os.path.join('examples','valid_raw_data','annotations','input.csv')
@pytest.mark.parametrize('file,result,labels',
    [(INP_CSV, [], ['set', 'recording_filename']),
     (INP_CSV, [0,1,2,3,4,5,6,7,8,9], []),
     (INP_CSV, [3,4,8,9], ['set']),
    ])
def test_find_lines_involved_in_overlaps(file, result, labels):
    
    df = pd.read_csv(file)
    truth = df.iloc[result]
    
    df_no_ov = df[find_lines_involved_in_overlap(df, 'range_onset', 'range_offset', labels)]
    
    pd.testing.assert_frame_equal(df_no_ov, truth, check_like = True)
    
    
    
    
    
    
    
    
