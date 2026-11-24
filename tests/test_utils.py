import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile

from ChildProject.projects import ChildProject
from ChildProject.utils import (
    Segment,
    TimeInterval,
    calculate_shift,
    df_to_printable,
    find_lines_involved_in_overlap,
    get_audio_duration,
    intersect_ranges,
    printable_unit_duration,
    read_wav,
    retry_func,
    series_to_datetime,
    time_intervals_intersect,
)

from conftest import EXAMPLES

RAW_VALID = EXAMPLES / "valid_raw_data" / "recordings" / "raw"
RAW_INVALID = EXAMPLES / "invalid_raw_data" / "recordings" / "raw"
MONO_4096 = RAW_VALID / "sound.wav"          # 4 s, mono, 4096 Hz
MONO_4096_B = RAW_VALID / "sound2.wav"       # 4 s, mono, 4096 Hz
STEREO_16K = RAW_INVALID / "test_1_20160918.wav"  # 4 s, stereo, 16 kHz
VIDEO = RAW_VALID / "video.mp4"              # 10.12 s container, h264 + stereo aac 44.1 kHz
CORRUPTED_MP3 = RAW_VALID / "unindexed.mp3"  # not decodable


def t(h, m=0, s=0):
    return datetime.datetime(1900, 1, 1, h, m, s)


# ---------------------------------------------------------------- Segment
def test_segment_length():
    assert Segment(3, 10).length() == 7


def test_segment_repr():
    assert repr(Segment(3, 10)) == "Segment([3, 10])"


# ---------------------------------------------------------------- df_to_printable
@pytest.mark.parametrize("delimiter", ["", "ab"])
def test_df_to_printable_rejects_delimiter_not_one_char(delimiter):
    with pytest.raises(ValueError, match="Invalid Delimiter"):
        df_to_printable(pd.DataFrame({"a": [1]}), delimiter=delimiter)


def test_df_to_printable_right_aligns_columns_and_appends_colored_index():
    df = pd.DataFrame({"a": ["x", "yyy"], "bb": [1, 22]}, index=["r1", "r2"])
    assert df_to_printable(df) == (
        "  x  1 \033[94mr1\033[0m\n"
        "yyy 22 \033[94mr2\033[0m\n"
    )


def test_df_to_printable_header_uses_column_names():
    df = pd.DataFrame({"long_name": ["v"]}, index=["r"])
    assert df_to_printable(df, header=True) == (
        "long_name\n"
        "        v \033[94mr\033[0m\n"
    )


def test_df_to_printable_custom_delimiter():
    df = pd.DataFrame({"a": [1], "b": [2]}, index=["r"])
    assert df_to_printable(df, delimiter=",", header=True) == "a,b\n1,2,\033[94mr\033[0m\n"


def test_df_to_printable_missing_values_become_empty():
    df = pd.DataFrame({"a": [np.nan]}, index=["r"])
    # empty value right-justified on the header width (1) + delimiter + index
    assert df_to_printable(df) == "  \033[94mr\033[0m\n"


# ---------------------------------------------------------------- printable_unit_duration
@pytest.mark.parametrize("duration,expected", [
    (0, "0ms"),
    (999, "999ms"),
    (1000, "1.0s"),
    (59_999, "60.0s"),
    (60_000, "1.0m"),
    (90_000, "1.5m"),
    (3_599_999, "60.0m"),
    (3_600_000, "1.0h"),
    (5_400_000, "1.5h"),
])
def test_printable_unit_duration(duration, expected):
    assert printable_unit_duration(duration) == expected


# ---------------------------------------------------------------- retry_func
class Flaky:
    def __init__(self, failures, exc=OSError):
        self.failures = failures
        self.calls = 0
        self.exc = exc

    def __call__(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        if self.calls <= self.failures:
            raise self.exc("boom")


def test_retry_func_calls_once_when_successful():
    func = Flaky(failures=0)
    retry_func(func, OSError, tries=3, a=1)
    assert func.calls == 1
    assert func.kwargs == {"a": 1}


def test_retry_func_retries_until_success():
    func = Flaky(failures=2)
    retry_func(func, OSError, tries=3)
    assert func.calls == 3


def test_retry_func_reraises_after_last_try():
    func = Flaky(failures=5)
    with pytest.raises(OSError):
        retry_func(func, OSError, tries=3)
    assert func.calls == 3


def test_retry_func_with_zero_tries_never_calls():
    func = Flaky(failures=0)
    assert retry_func(func, OSError, tries=0) is None
    assert func.calls == 0


def test_retry_func_does_not_retry_other_exceptions():
    func = Flaky(failures=5, exc=KeyError)
    with pytest.raises(KeyError):
        retry_func(func, OSError, tries=3)
    assert func.calls == 1


# ---------------------------------------------------------------- intersect_ranges
def _ranges(*pairs):
    return iter([Segment(a, b) for a, b in pairs])


@pytest.mark.parametrize("xs,ys,expected", [
    ([], [(0, 10)], []),
    ([(0, 10)], [], []),
    ([(0, 10)], [(5, 15)], [(5, 10)]),
    ([(0, 10)], [(10, 20)], []),  # touching ranges have an empty intersection
    ([(0, 10), (20, 30)], [(5, 25)], [(5, 10), (20, 25)]),
    ([(0, 100)], [(10, 20), (30, 40)], [(10, 20), (30, 40)]),
])
def test_intersect_ranges(xs, ys, expected):
    result = [(s.start, s.stop) for s in intersect_ranges(_ranges(*xs), _ranges(*ys))]
    assert result == expected


# ---------------------------------------------------------------- TimeInterval
def test_time_interval_drops_the_date():
    ti = TimeInterval(datetime.datetime(2022, 5, 3, 8, 0), datetime.datetime(2023, 1, 1, 9, 0))
    assert ti.start == t(8) and ti.stop == t(9)


def test_time_interval_length():
    assert TimeInterval(t(8), t(9, 30)).length() == datetime.timedelta(hours=1, minutes=30)


def test_time_interval_repr():
    assert repr(TimeInterval(t(8), t(9))) == "TimeInterval([1900-01-01 08:00:00, 1900-01-01 09:00:00])"


def test_time_interval_equality():
    assert TimeInterval(t(8), t(9)) == TimeInterval(t(8), t(9))
    assert not TimeInterval(t(8), t(9)) == TimeInterval(t(8), t(10))


# ---------------------------------------------------------------- time_intervals_intersect
@pytest.mark.parametrize("ti1,ti2,expected", [
    # overlap: [start2, stop1]
    ((t(8, 57), t(21, 4)), (t(10, 36), t(22, 1)), [(t(10, 36), t(21, 4))]),
    # second interval inside the first
    ((t(8), t(20)), (t(10), t(12)), [(t(10), t(12))]),
    # first interval inside the second
    ((t(10), t(12)), (t(8), t(20)), [(t(10), t(12))]),
    # overlap: [start1, stop2]
    ((t(10), t(20)), (t(8), t(12)), [(t(10), t(12))]),
    # second interval wraps around midnight: two intersections
    ((t(8, 57), t(22, 1)), (t(21, 4), t(10, 36)), [(t(8, 57), t(10, 36)), (t(21, 4), t(22, 1))]),
    # first interval wraps around midnight: two intersections
    ((t(22), t(2)), (t(1), t(23)), [(t(22), t(23)), (t(1), t(2))]),
    # both wrap around midnight
    ((t(8, 57), t(3, 1)), (t(21, 4), t(7, 36)), [(t(21, 4), t(3, 1))]),
    # disjoint
    ((t(8), t(10)), (t(12), t(14)), []),
    # touching intervals produce an empty (removed) intersection
    ((t(8), t(10)), (t(10), t(12)), []),
])
def test_time_intervals_intersect(ti1, ti2, expected):
    result = time_intervals_intersect(TimeInterval(*ti1), TimeInterval(*ti2))
    assert result == [TimeInterval(*e) for e in expected]


# ---------------------------------------------------------------- get_audio_duration
@pytest.mark.parametrize("file,expected", [
    (MONO_4096, 4.0),
    (STEREO_16K, 4.0),
    (VIDEO, 10.12),
])
def test_get_audio_duration_in_seconds(file, expected):
    duration = get_audio_duration(file)
    assert isinstance(duration, float)
    assert duration == pytest.approx(expected)


def test_get_audio_duration_missing_file_is_zero(tmp_path, capsys):
    assert get_audio_duration(tmp_path / "nope.wav") == 0
    assert "could not find file" in capsys.readouterr().out


def test_get_audio_duration_unreadable_file_is_zero(capsys):
    assert get_audio_duration(CORRUPTED_MP3) == 0
    assert "could not read duration" in capsys.readouterr().out


# ---------------------------------------------------------------- read_wav
def test_read_wav_mono():
    y, sr, channels = read_wav(MONO_4096, 0, 1)
    assert (sr, channels) == (4096, 1)
    assert y.shape == (4096,)


def test_read_wav_stereo():
    y, sr, channels = read_wav(STEREO_16K, 1, 2)
    assert (sr, channels) == (16000, 2)
    assert y.shape == (2, 32000)


def test_read_wav_on_video():
    with pytest.raises(soundfile.LibsndfileError):
        y, sr, channels = read_wav(VIDEO, 0, 1)


# ---------------------------------------------------------------- calculate_shift
def test_calculate_shift_identical_files_have_zero_divergence():
    score, size = calculate_shift(MONO_4096, MONO_4096, 0, 0, 4)
    assert score == 0
    assert size == 1639


def test_calculate_shift_identical_stereo_files_have_zero_divergence():
    score, size = calculate_shift(STEREO_16K, STEREO_16K, 0, 0, 2)
    assert score == 0
    assert size == 1600  # 2 channels * 2 s * 400 Hz


@pytest.mark.parametrize("f1,f2,score,size", [
    (MONO_4096, MONO_4096_B, 168.33635740995842, 1639),
    (MONO_4096, STEREO_16K, 47.00133204460144, 512),   # stereo shrunk to mono, rates aligned
    (MONO_4096_B, STEREO_16K, 163.95631432533264, 512),
])
def test_calculate_shift_known_values(f1, f2, score, size):
    result_score, result_size = calculate_shift(f1, f2, 0, 0, 4)
    assert result_score == pytest.approx(score)
    assert result_size == size


def test_calculate_shift_is_symmetric_with_channel_shrinking():
    forward = calculate_shift(MONO_4096, STEREO_16K, 0, 0, 4)
    backward = calculate_shift(STEREO_16K, MONO_4096, 0, 0, 4)
    assert forward == pytest.approx(backward)


@pytest.mark.parametrize("reverse", [False, True])
def test_calculate_shift_downsamples_only_the_higher_rate(tmp_path, reverse):
    """8192 Hz vs 4096 Hz: gcd is 4096, only the 8192 Hz signal is decimated"""
    signal, _ = soundfile.read(MONO_4096)
    doubled = tmp_path / "doubled.wav"
    soundfile.write(doubled, np.repeat(signal, 2), samplerate=8192)
    files = (doubled, MONO_4096) if reverse else (MONO_4096, doubled)
    score, size = calculate_shift(*files, 0, 0, 4)
    assert score == pytest.approx(0)
    assert size == 1639


def test_calculate_shift_incompatible_channel_counts(tmp_path):
    three_channels = tmp_path / "three.wav"
    soundfile.write(three_channels, np.zeros((16000, 3)), samplerate=16000)
    with pytest.raises(Exception, match="audios do not match"):
        calculate_shift(STEREO_16K, three_channels, 0, 0, 1)


# ---------------------------------------------------------------- find_lines_involved_in_overlap
@pytest.fixture
def ranges():
    return pd.DataFrame({
        "set": ["a", "a", "b", "b"],
        "rec": ["r1", "r2", "r1", "r1"],
        "range_onset": [0, 50, 100, 150],
        "range_offset": [100, 150, 200, 250],
    })


@pytest.mark.parametrize("labels,expected", [
    ([], [True, True, True, True]),
    (["set"], [True, True, True, True]),
    (["rec"], [False, False, True, True]),  # r1 lines 0 and 2 only touch
    (["set", "rec"], [False, False, True, True]),
])
def test_find_lines_involved_in_overlap(ranges, labels, expected):
    result = find_lines_involved_in_overlap(ranges, labels=labels)
    assert result.tolist() == expected


def test_find_lines_involved_in_overlap_custom_columns():
    df = pd.DataFrame({"segment_onset": [0, 10, 30], "segment_offset": [20, 25, 40]})
    result = find_lines_involved_in_overlap(df, "segment_onset", "segment_offset")
    assert result.tolist() == [True, True, False]


def test_find_lines_involved_in_overlap_on_example_index():
    df = pd.read_csv(EXAMPLES / "valid_raw_data" / "annotations" / "input.csv")
    assert not find_lines_involved_in_overlap(df, labels=["set", "recording_filename"]).any()
    assert find_lines_involved_in_overlap(df, labels=["set"]).tolist() == [i in [3, 4, 8, 9] for i in range(len(df))]


# ---------------------------------------------------------------- series_to_datetime
TIMES = pd.Series(["3:12", "04:14", "12:19:21", "32:77", "NA"])
DATES = pd.Series(["2022-01-23"] * 5)


def test_series_to_datetime_time_only():
    result = series_to_datetime(TIMES, ChildProject.RECORDINGS_COLUMNS, "start_time")
    expected = pd.Series([t(3, 12), t(4, 14), t(12, 19, 21), pd.NaT, pd.NaT])
    pd.testing.assert_series_equal(result, expected, check_dtype=False)


def test_series_to_datetime_with_date():
    result = series_to_datetime(TIMES, ChildProject.RECORDINGS_COLUMNS, "start_time",
                                DATES, ChildProject.RECORDINGS_COLUMNS, "date_iso")
    expected = pd.Series([datetime.datetime(2022, 1, 23, 3, 12), datetime.datetime(2022, 1, 23, 4, 14),
                          datetime.datetime(2022, 1, 23, 12, 19, 21), pd.NaT, pd.NaT])
    pd.testing.assert_series_equal(result, expected, check_dtype=False)


def test_series_to_datetime_keeps_index():
    times = pd.Series(["3:12"], index=[42])
    result = series_to_datetime(times, ChildProject.RECORDINGS_COLUMNS, "start_time")
    assert result.index.tolist() == [42]
