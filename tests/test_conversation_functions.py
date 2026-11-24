"""conversation features (ChildProject.pipelines.conversationFunctions)"""
import math

import pandas as pd
import pytest

import ChildProject.pipelines.conversationFunctions as cf
from ChildProject.pipelines.conversationFunctions import RESERVED, conversationFunction

from conftest import TEST_DATA


@pytest.fixture
def segments():
    segments = pd.read_csv(TEST_DATA / "csv.csv").dropna(subset=["speaker_type"])
    segments["voc_duration"] = segments["segment_offset"] - segments["segment_onset"]
    return segments


@pytest.mark.parametrize("function,parameters,expected", [
    (cf.who_initiated, {}, "CHI"),
    (cf.who_finished, {}, "MAL"),
    (cf.participants, {}, "CHI/OCH/FEM/MAL"),
    (cf.voc_total_dur, {}, 15034),
    (cf.is_speaker, {"speaker": "XXX"}, False),
    (cf.is_speaker, {"speaker": "OCH"}, True),
    (cf.voc_speaker_count, {"speaker": "CHI"}, 1),
    (cf.voc_speaker_count, {"speaker": "OCH"}, 3),
    (cf.voc_speaker_dur, {"speaker": "MAL"}, 3924),
    (cf.voc_speaker_dur, {"speaker": "FEM"}, 3459),
    (cf.voc_dur_contribution, {"speaker": "FEM"}, 3459 / 15034),
    (cf.voc_dur_contribution, {"speaker": "OCH"}, 6794 / 15034),
    (cf.assign_conv_type, {}, "multiparty"),
])
def test_conversation_features(segments, function, parameters, expected):
    assert function(segments, **parameters) == pytest.approx(expected)


@pytest.mark.parametrize("speakers,expected", [
    (["FEM", "MAL"], "overheard"),
    (["CHI", "FEM"], "dyadic_FEM"),
    (["CHI", "MAL"], "dyadic_MAL"),
    (["CHI", "OCH"], "peer"),
    (["CHI", "FEM", "MAL"], "parent"),
    (["CHI", "OCH", "FEM"], "triadic_FEM"),
    (["CHI", "OCH", "MAL"], "triadic_MAL"),
    (["CHI", "OCH", "FEM", "MAL"], "multiparty"),
])
def test_assign_conv_type(speakers, expected):
    assert cf.assign_conv_type(pd.DataFrame({"speaker_type": speakers})) == expected


def test_assign_conv_type_child_alone_is_undefined():
    assert math.isnan(cf.assign_conv_type(pd.DataFrame({"speaker_type": ["CHI", "CHI"]})))


@pytest.mark.parametrize("function", [cf.is_speaker, cf.voc_speaker_count, cf.voc_speaker_dur, cf.voc_dur_contribution])
def test_speaker_features_require_speaker(segments, function):
    with pytest.raises(ValueError, match="needs an argument <speaker>"):
        function(segments)


@pytest.mark.parametrize("reserved", sorted(RESERVED))
def test_conversation_function_rejects_reserved_arguments(reserved):
    with pytest.raises(ValueError, match="you cannot use reserved keywords"):
        @conversationFunction({reserved})
        def fake(segments, **kwargs):
            return 0


def test_conversation_function_keeps_name_and_passes_kwargs(segments):
    @conversationFunction({"factor"})
    def scaled_count(segments, **kwargs):
        return segments.shape[0] * kwargs["factor"]

    assert scaled_count.__name__ == "scaled_count"
    assert scaled_count(segments, factor=2) == 2 * segments.shape[0]
