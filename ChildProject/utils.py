import os
import librosa
from scipy.fftpack import fft, ifft
import numpy as np

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
