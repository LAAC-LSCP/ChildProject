import os
from datetime import datetime
from scipy.fftpack import fft, ifft
import numpy as np
import pandas as pd

def path_is_parent(parent_path: str, child_path: str):
    # Smooth out relative path names, note: if you are concerned about symbolic links, you should use os.path.realpath too
    parent_path = os.path.abspath(parent_path)
    child_path = os.path.abspath(child_path)

    # Compare the common path of the parent and child path with the common path of just the parent path. Using the commonpath method on just the parent path will regularise the path name in the same way as the comparison that deals with both paths, removing any trailing path separator
    return os.path.commonpath([parent_path]) == os.path.commonpath(
        [parent_path, child_path]
    )


class Segment:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def length(self):
        return self.stop - self.start

    def __repr__(self):
        return "Segment([{}, {}])".format(self.start, self.stop)


def intersect_ranges(xs, ys):
    # Try to get the first range in each iterator:
    try:
        x, y = next(xs), next(ys)
    except StopIteration:
        return

    while True:
        # Yield the intersection of the two ranges, if it's not empty:
        intersection = Segment(max(x.start, y.start), min(x.stop, y.stop))
        if intersection.length() > 0:
            yield intersection

        # Try to increment the range with the earlier stopping value:
        try:
            if x.stop <= y.stop:
                x = next(xs)
            else:
                y = next(ys)
        except StopIteration:
            return

class TimeInterval:
    def __init__(self, start : datetime, stop : datetime):
        #remove the day/month/year component
        self.start = start.replace(year=1900, month=1, day=1)
        self.stop = stop.replace(year=1900, month=1, day=1)

    def length(self):
        return self.stop - self.start

    def __repr__(self):
        return "TimeInterval([{}, {}])".format(self.start, self.stop)
    
    def __eq__(self, other):
        return self.start == other.start and self.stop == other.stop
    
def time_intervals_intersect(ti1 : TimeInterval, ti2 : TimeInterval):
    """
    given 2 time intervals (those do not take in consideration days, only time in the day), return an array of new interval(s) representing the intersections of the original ones.
    Examples
    1. time_intervals_intersect( TimeInterval( datetime(1900,1,1,8,57), datetime(1900,1,1,21,4)), TimeInterval( datetime(1900,1,1,10,36), datetime(1900,1,1,22,1))) => [TimeInterval(10:36 , 21:04)]
    2. time_intervals_intersect( TimeInterval( datetime(1900,1,1,8,57), datetime(1900,1,1,22,1)), TimeInterval( datetime(1900,1,1,21,4), datetime(1900,1,1,10,36))) => [TimeInterval(08:57 , 10:36),TimeInterval(21:04 , 22:01)]
    
    :param ti1: first interval
    :param ti2: second interval
    :type ti1: TimeInterval
    :type ti2: TimeInterval
    """
    #The calculation and boolean evaluation is done that way to optimize the process, those expressions were obtained using a Karnaugh table. Given the relations between the different start and ending times, the boolean relations used below gives the correct intervals
    a = ti1.start <= ti1.stop
    b = ti2.start <= ti2.stop
    c = ti1.stop <= ti2.stop
    d = ti1.start <= ti2.start
    e = ti1.start <= ti2.stop
    f = ti2.start <= ti1.stop
    r = []
    #case where correct resulting interval is [start of the 2nd interval : end of the 1st interval]
    if c and (d and (not e or f) or not e and f) or d and not e and f : r = [TimeInterval(ti2.start,ti1.stop)]
    #case where correct resulting interval is [start of the 2nd interval : end of the 2nd interval]
    elif not c and (d and (b or not a) or not a and b) or not a and b and d : r = [ti2]
    #case where correct resulting interval is [start of the 1st interval : end of the 2nd interval]
    elif not c and (not d and (not e and not f or e) or e and not f) or not d and e and not f : r = [TimeInterval(ti1.start,ti2.stop)]
    #case where correct resulting interval is [start of the 1st interval : end of the 1st interval]
    elif c and (not d and (not a and not b or a) or a and not b) or a and not b and not d : r = [ti1]
    # !! here the expression was simplified because previous statements should already have caught their relevant cases (ie this statement should always be last or changed)
    #case where correct resulting interval is [start of the 1st interval : end of the 2nd interval] U [start of the 2nd interval : end of the 1st interval]
    elif not a and (not b or e) or d and e and f : r = [TimeInterval(ti1.start,ti2.stop),TimeInterval(ti2.start,ti1.stop)]
    
    #remove the intervals having equal values (3:00 to 3:00)
    i = 0
    while i < len(r):
        if r[i].start == r[i].stop : r.pop(i)
        else : i += 1
    return r

def get_audio_duration(filename):
    import sox

    if not os.path.exists(filename):
        return 0

    duration = 0
    try:
        duration = sox.file_info.duration(filename)
    except:
        pass

    return duration

#reads a wav file for a given start point and duration (both in seconds)
def read_wav(filename, start_s, length_s):
    import librosa
    #we use librosa because it supports more codecs and is less likely to crash on an unsual encoding
    y,sr = librosa.load(filename, sr=None,mono=False, offset=start_s, duration = length_s)
    channels = 1 if len(y.shape) == 1 else y.shape[0]

    return y, sr, channels

#computes a list of similarity scores, each value being for 1 frame of audio
#returns the most differing value
def calculate_shift(file1, file2, start1, start2, interval, correlation_output = None):
    ref, ref_rate, ref_chan = read_wav(file1, start1, interval)
    test, test_rate, test_chan = read_wav(file2, start2, interval)

    #when sampling rate is different, look for a downsampled rate that can be used
    if ref_rate != test_rate:
        from math import gcd
        new_rate = gcd(ref_rate,test_rate)
        print('WARNING : sampling rates do not match between audios ({}Hz and {}Hz), attempting to downsample to {}Hz'.format(ref_rate,test_rate,new_rate))
        if ref_rate > new_rate : #downsample if needed
            ref = ref[::int(ref_rate/new_rate)]
            ref_rate = new_rate
        if test_rate > new_rate : #downsample if needed
            test = test[::int(test_rate/new_rate)]
            test_rate = new_rate
        
    if ref_chan != test_chan: #if different number of channels, shrink if possible
        print('WARNING : different number of channels, attempting to compress channels to carry on with analysis')
        if ref_chan == 1 and test_chan > 1 :
            test = np.mean(test,axis=0)
            test_chan = 1
            print('{} was shrunk to mono channel for the analysis, it has a higher level of information than {}'.format(file2,file1))
        elif ref_chan > 1 and test_chan == 1:
            ref = np.mean(ref,axis=0)
            ref_chan = 1
            print('{} was shrunk to mono channel for the analysis, it has a higher level of information than {}'.format(file1,file2))
        else:
            raise Exception('audios do not match, {} has {} channel(s) while {} has {}'.format(file1,ref_chan,file2,test_chan))
        
    #in case of multiple channels, reshape to be 1D array (they should have the same number of channels at this point)
    if ref_chan > 1:
        np.reshape(ref,ref_chan * ref.shape[1])
        np.reshape(test,test_chan * test.shape[1])

    sampling_rate = ref_rate

    #downsample to save computation time
    downsampled_rate = 400
    ref = ref[::int(sampling_rate/downsampled_rate)]
    test = test[::int(sampling_rate/downsampled_rate)]
    
    # straight up difference of the audio signal averaged over the 2 segments analysed
    # times 1000 is arbitrary, just to have an easily readable and comparable score output
    res = np.abs(ref - test).sum() * 1000 /(len(ref))

    return res,len(ref)

def series_to_datetime(time_series, time_index_list, time_column_name:str, date_series = None, date_index_list = None, date_column_name = None):
    """
    returns a series of datetimes from a series of str. Using pd.to_datetime on all the formats \
    listed for a specific column name in an index consisting of IndexColumn items. \
    To have the date included and not only time), one can use a second series for date, \
    with also the corresponding index and column
    
    :param time_series: pandas series of strings to transform into datetime (can contain NA value => NaT datetime), if date_series is given, time_series should only have the time
    :type time_series: pandas.Series
    :param time_index_list: list of index to use where the column wanted is present
    :type time_index_list: List[IndexColumn]
    :param time_column_name: name of the IndexColumn to use (IndexColumn.name value) for accepted formats
    :type time_column_name: str
    :param date_series: pandas series of strings to transform into the date component of datetime (can contain NA value)
    :type date_series: pandas.Series
    :param date_index_list: list of index to use where the column wanted is present
    :type date_index_list: List[IndexColumn]
    :param date_column_name: name of the IndexColumn to use (IndexColumn.name value) for accepted formats for dates
    :type date_column_name: str
    :return: series with dtype datetime containing the converted datetimes
    :rtype: pandas.Series
    """
    time_formats = next(x for x in time_index_list if x.name==time_column_name).datetime
    series = pd.Series(np.nan, index=np.arange(time_series.shape[0]) , dtype='datetime64[ns]')
    if date_series is not None:
        time_sr = date_series + ' ' + time_series
        date_formats = next(x for x in date_index_list if x.name==date_column_name).datetime
        for frmt in time_formats:
            for dfrmt in date_formats:
                series = series.fillna(pd.to_datetime(time_sr, format="{} {}".format(dfrmt,frmt), errors="coerce"))
    else:
        time_sr = time_series.copy()
        for frmt in time_formats:
            series = series.fillna(pd.to_datetime(time_sr, format=frmt, errors="coerce"))
    return series
    