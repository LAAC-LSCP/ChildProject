#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 11:23:05 2022

@author: lpeurey
"""
from ChildProject.projects import ChildProject
from ChildProject.utils import series_to_datetime, time_intervals_intersect, intersect_ranges, TimeInterval
import pandas as pd
import datetime

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
    
    pd.testing.assert_series_equal(converted_time, truth)
    
    truth = pd.Series([datetime.datetime(2022,1,23,3,12,0,0),
                       datetime.datetime(2022,1,23,4,14,0,0),
                       datetime.datetime(2022,1,23,12,19,21,0),
                       pd.NaT,
                       ])
    
    #convert to datetime using the formats for start_time and date_iso listed in the project RECORDINGS_COLUMNS.Index(name == 'start_time') and Index(name == 'date_iso')
    converted_time = series_to_datetime(only_time, project.RECORDINGS_COLUMNS, 'start_time', only_date, project.RECORDINGS_COLUMNS, 'date_iso')
    
    pd.testing.assert_series_equal(converted_time, truth)
    
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
        