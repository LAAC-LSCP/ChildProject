"""unit tests of the metric functions and their decorators (ChildProject.pipelines.metricsFunctions)"""
import math

import numpy as np
import pandas as pd
import pytest

import ChildProject.pipelines.metricsFunctions as mf
from ChildProject.pipelines.metricsFunctions import RESERVED, metricFunction, peak_hour_metric, per_hour_metric

HOUR = 3_600_000


@pytest.fixture
def vtc():
    """VTC/VCM/ALICE-like segments, sorted by onset"""
    df = pd.DataFrame([
        ("CHI", 0, 1000, "C", 0, 0, 0),
        ("CHI", 1500, 2000, "N", 0, 0, 0),
        ("FEM", 2100, 2600, np.nan, 3, 4, 9),
        ("CHI", 3000, 3300, "Y", 0, 0, 0),
        ("MAL", 4000, 5000, np.nan, 2, 2, 5),
        ("OCH", 6000, 6100, np.nan, 0, 0, 0),
    ], columns=["speaker_type", "segment_onset", "segment_offset", "vcm_type", "words", "syllables", "phonemes"])
    df["duration"] = df["segment_offset"] - df["segment_onset"]
    df["set"] = "vtc"
    return df


@pytest.fixture
def lena():
    """LENA its-like segments"""
    return pd.DataFrame({
        "set": "its",
        "speaker_type": ["CHI", "CHI", "FEM"],
        "segment_onset": [0, 1000, 2000],
        "segment_offset": [500, 1500, 2500],
        "utterances_count": [2, 1, 0],
        "cries": ["[{'start': 0, 'end': 1}]", "[{'start': 5, 'end': 6}, {'start': 7, 'end': 8}]", "[]"],
        "vfxs": ["[]", "[{'start': 0, 'end': 1}]", "[]"],
        "child_cry_vfx_len": [100, 50, 0],
        "utterances_length": [900, 400, 0],
        "lena_conv_turn_type": ["TIFR", "TIMR", "NT"],
    })


def call(metric, segments, duration=HOUR, **kwargs):
    return getattr(mf, metric)(segments, duration, **kwargs)


# ============================================================ decorators
@pytest.mark.parametrize("reserved", sorted(RESERVED))
def test_metric_function_rejects_reserved_arguments(reserved):
    with pytest.raises(ValueError, match="you cannot use reserved keywords"):
        @metricFunction({reserved}, set())
        def fake(segments, duration, **kwargs):
            return 0


def test_metric_function_requires_its_arguments(vtc):
    with pytest.raises(ValueError, match="voc_speaker metric needs an argument <speaker>"):
        call("voc_speaker", vtc)


def test_metric_function_requires_columns(vtc):
    with pytest.raises(ValueError, match=r"The given set <vtc> does not have the required column\(s\)"):
        call("lena_CVC", vtc)


def test_metric_function_requires_one_of_alternative_columns(vtc):
    with pytest.raises(ValueError, match=" or "):
        call("cry_voc_speaker", vtc.drop(columns=["vcm_type"]), speaker="CHI")


def test_metric_function_name_replaces_arguments(vtc):
    assert call("voc_speaker", vtc, speaker="CHI")[0] == "voc_chi"


def test_metric_function_explicit_name(vtc):
    assert call("voc_speaker", vtc, speaker="CHI", name="kids")[0] == "kids"


def test_metric_function_empty_name_is_ignored(vtc):
    assert call("voc_speaker", vtc, speaker="CHI", name="")[0] == "voc_chi"


def test_metric_function_default_name(vtc):
    @metricFunction({"speaker"}, {"speaker_type"}, default_name="my_speaker_count")
    def count(segments, duration, **kwargs):
        return segments.shape[0]

    assert count(vtc, HOUR, speaker="FEM") == ("my_fem_count", 6)


@pytest.mark.parametrize("metric,empty", [("voc_speaker", 0), ("avg_voc_dur_speaker", None)])
def test_metric_function_empty_segments_but_annotated(metric, empty):
    name, value = call(metric, pd.DataFrame(), HOUR, speaker="CHI")
    if empty is None:
        assert math.isnan(value)
    else:
        assert value == empty


def test_metric_function_unit_not_annotated_is_none():
    assert call("voc_speaker", pd.DataFrame(), 0, speaker="CHI") == ("voc_chi", None)


def test_per_hour_metric_scales_by_duration(vtc):
    assert call("voc_speaker_ph", vtc, HOUR // 2, speaker="CHI") == ("voc_chi_ph", 6)


def test_per_hour_metric_naming():
    @per_hour_metric()
    def thing(segments, duration, **kwargs):
        """doc"""
        return 1

    assert thing.__name__ == "thing_ph"
    assert thing.__doc__.endswith("This value is a 'per hour' value.")


def test_peak_hour_metric(vtc):
    # periods of 2 s over 6 s: CHI counts are 2, 1, 0
    assert call("peak_voc_speaker", vtc, 6000, speaker="CHI", period_time=2000) == ("peak_voc_chi", 2)


def test_peak_hour_metric_default_period_is_one_hour(vtc):
    assert call("peak_voc_speaker", vtc, 2 * HOUR, speaker="CHI")[1] == 3


def test_peak_hour_metric_without_complete_period(vtc):
    assert math.isnan(call("peak_voc_speaker", vtc, 1000, speaker="CHI", period_time=2000)[1])


def test_peak_hour_metric_empty_periods_use_empty_value(vtc):
    assert call("peak_voc_speaker", vtc, 6000, speaker="OCH", period_time=2000)[1] == 0


def test_peak_hour_metric_naming():
    assert mf.peak_voc_speaker.__name__ == "peak_voc_speaker"
    assert mf.peak_voc_speaker.__doc__.startswith("Computing the peak for 1h")


def test_peak_hour_metric_does_not_modify_segments(vtc):
    before = vtc.copy()
    call("peak_voc_speaker", vtc, 6000, speaker="CHI", period_time=2000)
    pd.testing.assert_frame_equal(vtc, before)


# ============================================================ vocalization metrics
@pytest.mark.parametrize("metric,speaker,expected", [
    ("voc_speaker", "CHI", 3),
    ("voc_speaker", "FEM", 1),
    ("voc_dur_speaker", "CHI", 1800),
    ("avg_voc_dur_speaker", "CHI", 600),
    ("std_voc_dur_speaker", "CHI", math.sqrt(130000)),
    ("wc_speaker", "FEM", 3),
    ("sc_speaker", "FEM", 4),
    ("pc_speaker", "FEM", 9),
    ("cry_voc_speaker", "CHI", 1),
    ("cry_voc_dur_speaker", "CHI", 300),
    ("avg_cry_voc_dur_speaker", "CHI", 300),
    ("can_voc_speaker", "CHI", 1),
    ("can_voc_dur_speaker", "CHI", 1000),
    ("avg_can_voc_dur_speaker", "CHI", 1000),
    ("non_can_voc_speaker", "CHI", 1),
    ("non_can_voc_dur_speaker", "CHI", 500),
    ("avg_non_can_voc_dur_speaker", "CHI", 500),
])
def test_speaker_metrics(vtc, metric, speaker, expected):
    name, value = call(metric, vtc, speaker=speaker)
    assert name == metric.replace("speaker", speaker.lower())
    assert value == pytest.approx(expected)


@pytest.mark.parametrize("metric", ["std_cry_voc_dur_speaker", "std_can_voc_dur_speaker", "std_non_can_voc_dur_speaker"])
def test_std_of_a_single_vocalization_is_undefined(vtc, metric):
    assert math.isnan(call(metric, vtc, speaker="CHI")[1])


@pytest.mark.parametrize("metric,expected", [
    ("avg_cry_voc_dur_speaker", 0), ("avg_can_voc_dur_speaker", 0), ("avg_non_can_voc_dur_speaker", 0),
])
def test_average_without_matching_vocalization_is_zero(vtc, metric, expected):
    assert call(metric, vtc, speaker="FEM")[1] == expected


@pytest.mark.parametrize("metric,expected", [("wc_adu", 5), ("sc_adu", 6), ("pc_adu", 14)])
def test_adult_metrics(vtc, metric, expected):
    assert call(metric, vtc) == (metric, expected)


@pytest.mark.parametrize("metric,expected", [("lp_n", 2 / 3), ("cp_n", 1 / 2), ("lp_dur", 1500 / 1800), ("cp_dur", 1000 / 1500)])
def test_proportions_from_vcm(vtc, metric, expected):
    assert call(metric, vtc)[1] == pytest.approx(expected)


@pytest.mark.parametrize("metric", ["lp_n", "cp_n", "lp_dur", "cp_dur"])
def test_proportions_without_child_vocalizations_are_nan(vtc, metric):
    assert math.isnan(call(metric, vtc[vtc["speaker_type"] != "CHI"])[1])


@pytest.mark.parametrize("metric", ["lp_n", "lp_dur"])
def test_proportions_require_vcm_or_lena_columns(vtc, metric):
    with pytest.raises(ValueError, match="does not have the necessary columns"):
        call(metric, vtc.drop(columns=["vcm_type"]))


def test_lp_dur_error_message_names_the_set(vtc):
    with pytest.raises(ValueError) as excinfo:
        call("lp_dur", vtc.drop(columns=["vcm_type"]))
    assert "{}" not in str(excinfo.value)


# ============================================================ LENA metrics
@pytest.mark.parametrize("metric,kwargs,expected", [
    ("lena_CVC", {}, 3),
    ("lena_CTC", {}, 2),
    ("cry_voc_speaker", {"speaker": "CHI"}, 3),
    ("cry_voc_dur_speaker", {"speaker": "CHI"}, 150),
    ("avg_cry_voc_dur_speaker", {"speaker": "CHI"}, 50),
    ("std_cry_voc_dur_speaker", {"speaker": "CHI"}, math.sqrt(1875)),
    ("lp_n", {}, 3 / 7),
    ("lp_dur", {}, 1300 / 1450),
])
def test_lena_metrics(lena, metric, kwargs, expected):
    assert call(metric, lena, **kwargs)[1] == pytest.approx(expected)


@pytest.mark.parametrize("metric", ["lp_n", "lp_dur"])
def test_lena_proportions_without_child_vocalizations(lena, metric):
    assert math.isnan(call(metric, lena[lena["speaker_type"] == "FEM"])[1])


# ============================================================ conversational turns
def test_simple_ctc(vtc):
    assert call("simple_CTC", vtc) == ("simple_CTC", 3)


def test_simple_ctc_max_interval(vtc):
    assert call("simple_CTC", vtc, max_interval=500)[1] == 2


def test_simple_ctc_min_delay(vtc):
    assert call("simple_CTC", vtc, min_delay=700)[1] == 2


def test_simple_ctc_custom_interlocutors(vtc):
    # only FEM <-> MAL turns count: none are adjacent
    assert call("simple_CTC", vtc, interlocutors_1=("FEM",), interlocutors_2=("MAL",))[1] == 0


def test_simple_ctc_overlapping_groups(vtc):
    assert call("simple_CTC", vtc, interlocutors_1=("CHI", "FEM"), interlocutors_2=("FEM", "MAL"))[1] == 3


def test_simple_ctc_without_speakers(vtc):
    assert call("simple_CTC", vtc.assign(speaker_type="SPEECH"))[1] == 0


def test_simple_ctc_per_hour(vtc):
    assert call("simple_CTC_ph", vtc, HOUR * 3)[1] == pytest.approx(1)
