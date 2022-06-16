import os
import wave
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
    fp = wave.open(filename)
    samples = fp.getnframes()
    sampling_rate = fp.getframerate()
    audio = fp.readframes(samples)

    audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16, offset = int(start_s*sampling_rate), count=int(length_s*sampling_rate+1))
    audio_as_np_float32 = audio_as_np_int16.astype(np.float32)
    
    max_int16 = 2**15
    audio_normalised = audio_as_np_float32 / max_int16

    return audio_normalised, sampling_rate

#computes a list of similarity scores, each value being for 1 frame of audio
#returns the most differing value
def calculate_shift(file1, file2, start1, start2, interval, correlation_output = None):
    ref, ref_rate = read_wav(file1, start1, interval)
    test, test_rate = read_wav(file2, start2, interval)

    if ref_rate != test_rate:
        raise Exception('audios do not match')

    sampling_rate = ref_rate

    downsampled_rate = 400
    ref = ref[::int(sampling_rate/downsampled_rate)]
    test = test[::int(sampling_rate/downsampled_rate)]

    ref_padded = np.concatenate((np.zeros(len(ref)-1), ref))
    test_padded = np.concatenate((test, np.zeros(len(ref)-len(test)+len(ref)-1)))

    ref_fft = fft(ref_padded)
    test_fft = fft(test_padded)

    cross_spectral_density = ref_fft*np.conj(test_fft)
    cross_correlations = ifft(cross_spectral_density)

    mag_cross_correlations = np.abs(cross_correlations)

    if correlation_output:
        np.savetxt(correlation_output, mag_cross_correlations, delimiter = ' ')

    #take the highest differing frame score
    shift = np.argmax(mag_cross_correlations) - max(len(ref),len(test))

    return shift/downsampled_rate
