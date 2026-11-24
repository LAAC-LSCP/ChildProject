"""unit tests of the derivators (ChildProject.pipelines.derivations); their use through
AnnotationManager.derive_annotations is tested in test_annotations_derive_merge.py"""
import numpy as np
import pandas as pd
import pytest

from ChildProject.annotations import AnnotationManager
from ChildProject.pipelines.derivations import (
    DERIVATIONS,
    AcousticDerivator,
    ConversationDerivator,
    CVADerivator,
    Derivator,
    RemoveOverlapsDerivator,
    RuntimeDerivator,
)

from conftest import EXAMPLES, TRUTH

VTC_SEGMENTS = EXAMPLES / "valid_raw_data" / "annotations" / "vtc_present" / "converted" / "sound2_0_4000.csv"
DERIVATIONS_TRUTH = TRUTH / "derivations"


@pytest.fixture
def vtc():
    return pd.read_csv(VTC_SEGMENTS)


def segments(rows):
    """rows of (speaker_type, onset, offset)"""
    return pd.DataFrame(rows, columns=["speaker_type", "segment_onset", "segment_offset"])


# ============================================================ registry / base classes
def test_derivations_registry():
    assert DERIVATIONS == {"acoustics": AcousticDerivator, "conversations": ConversationDerivator,
                           "remove-overlaps": RemoveOverlapsDerivator, "cva": CVADerivator}


def test_derivator_is_abstract():
    with pytest.raises(TypeError):
        Derivator()


@pytest.mark.parametrize("cls,media_types", [(Derivator, None), (AcousticDerivator, ["audio"]),
                                             (ConversationDerivator, None), (RemoveOverlapsDerivator, None),
                                             (CVADerivator, None)])
def test_media_types(cls, media_types):
    assert cls.MEDIA_TYPES == media_types


def test_runtime_derivator_calls_function_with_project_metadata_segments(vtc):
    calls = []
    derivator = RuntimeDerivator(lambda project, metadata, df: calls.append((project, metadata)) or df.head(1))
    result = derivator.derive("project", {"a": 1}, vtc)
    assert calls == [("project", {"a": 1})]
    assert result.shape[0] == 1


def test_runtime_derivator_has_no_metadata():
    assert RuntimeDerivator(lambda *a: None).get_auto_metadata(None, "in", "out") == {}


# ============================================================ get_auto_metadata
@pytest.fixture
def sets_am(am):
    am.sets = pd.DataFrame({"segmentation_type": ["permissive"], "has_speaker_type": ["Y"]}, index=["in"])
    return am


@pytest.fixture
def bare_am(am):
    am.sets = pd.DataFrame(index=["in"])
    return am


def test_acoustic_metadata(sets_am):
    assert AcousticDerivator(profile="p", target_sr=8000).get_auto_metadata(sets_am, "in", "out") == {
        "segmentation": "in", "has_acoustics": "Y", "parameters": {"profile": "p", "target_sr": 8000},
        "segmentation_type": "permissive", "has_speaker_type": "Y"}


def test_conversation_metadata(sets_am):
    meta = ConversationDerivator(max_interval=1000, min_delay=10).get_auto_metadata(sets_am, "in", "out")
    assert meta["segmentation"] == "in" and meta["has_interactions"] == "Y"
    assert meta["parameters"]["max_interval"] == 1000 and meta["parameters"]["min_delay"] == 10
    assert (meta["segmentation_type"], meta["has_speaker_type"]) == ("permissive", "Y")


def test_remove_overlaps_metadata(sets_am):
    assert RemoveOverlapsDerivator(speakers=["CHI"]).get_auto_metadata(sets_am, "in", "out") == {
        "segmentation": "out", "segmentation_type": "restrictive", "parameters": {"speakers": ["CHI"]},
        "has_speaker_type": "Y"}


def test_cva_metadata(sets_am):
    assert CVADerivator(iti=100, scenario="P").get_auto_metadata(sets_am, "in", "out") == {
        "segmentation": "out", "has_addressee": "Y", "parameters": {"iti": 100, "scenario": "P"},
        "segmentation_type": "permissive", "has_speaker_type": "Y"}


@pytest.mark.parametrize("derivator", [AcousticDerivator(), ConversationDerivator(), RemoveOverlapsDerivator(), CVADerivator()])
def test_metadata_without_set_information(bare_am, derivator):
    meta = derivator.get_auto_metadata(bare_am, "in", "out")
    assert "has_speaker_type" not in meta


# ============================================================ conversations
def test_conversation_derivator_matches_truth(vtc):
    result = ConversationDerivator().derive(None, {}, vtc)
    pd.testing.assert_frame_equal(result, pd.read_csv(DERIVATIONS_TRUTH / "conversations.csv"), check_dtype=False)


def test_conversation_derivator_turns_and_conversations():
    result = ConversationDerivator().derive(None, {}, segments([
        ("CHI", 0, 100), ("FEM", 200, 300), ("CHI", 10000, 10100), ("MAL", 10200, 10300)]))
    assert result["is_CT"].tolist() == [False, True, False, True]
    assert result["conv_count"].tolist() == [1, 1, 2, 2]


def test_conversation_derivator_ignores_unknown_speakers():
    result = ConversationDerivator().derive(None, {}, segments([("CHI", 0, 100), ("SPEECH", 150, 160), ("FEM", 200, 300)]))
    assert result["speaker_type"].tolist() == ["CHI", "FEM"]
    assert result["is_CT"].tolist() == [False, True]


def test_conversation_derivator_max_interval():
    result = ConversationDerivator(max_interval=50).derive(None, {}, segments([("CHI", 0, 100), ("FEM", 200, 300)]))
    assert result["is_CT"].tolist() == [False, False]


def test_conversation_derivator_min_delay():
    result = ConversationDerivator(min_delay=100).derive(None, {}, segments([("CHI", 0, 100), ("FEM", 50, 300)]))
    assert result["is_CT"].tolist() == [False, False]


def test_conversation_derivator_custom_interactions():
    interactions = {"CHI": {"FEM"}, "FEM": {"CHI"}, "MAL": set()}
    result = ConversationDerivator(interactions=interactions).derive(None, {}, segments([
        ("CHI", 0, 100), ("MAL", 200, 300), ("FEM", 400, 500)]))
    assert result["is_CT"].tolist() == [False, False, False]


def test_conversation_derivator_without_speakers():
    result = ConversationDerivator().derive(None, {}, segments([("SPEECH", 0, 100)]))
    assert result.shape[0] == 0
    assert set(result.columns) == {"segment_onset", "segment_offset", "raw_filename"}


# ============================================================ remove overlaps
def test_remove_overlaps_matches_truth(vtc):
    result = RemoveOverlapsDerivator().derive(None, {}, vtc)
    pd.testing.assert_frame_equal(result, pd.read_csv(DERIVATIONS_TRUTH / "remove-overlaps.csv"), check_dtype=False)


def _spans(df):
    return df[["speaker_type", "segment_onset", "segment_offset"]].values.tolist()


def test_remove_overlaps_splits_around_contained_overlap():
    result = RemoveOverlapsDerivator().derive(None, {}, segments([("CHI", 0, 1000), ("FEM", 500, 700)]))
    assert _spans(result) == [["CHI", 0, 500], ["CHI", 700, 1000]]


def test_remove_overlaps_trims_partial_overlaps():
    result = RemoveOverlapsDerivator().derive(None, {}, segments([("CHI", 0, 1000), ("FEM", 800, 1200)]))
    assert _spans(result) == [["CHI", 0, 800], ["FEM", 1000, 1200]]


def test_remove_overlaps_squashes_chained_overlaps():
    result = RemoveOverlapsDerivator().derive(None, {}, segments([
        ("CHI", 0, 1000), ("FEM", 100, 300), ("MAL", 200, 400), ("OCH", 600, 700)]))
    assert _spans(result) == [["CHI", 0, 100], ["CHI", 400, 600], ["CHI", 700, 1000]]


def test_remove_overlaps_keeps_isolated_segments():
    result = RemoveOverlapsDerivator().derive(None, {}, segments([("CHI", 0, 100), ("FEM", 200, 300)]))
    assert _spans(result) == [["CHI", 0, 100], ["FEM", 200, 300]]


def test_remove_overlaps_restricts_speakers():
    result = RemoveOverlapsDerivator(speakers=["CHI"]).derive(None, {}, segments([("CHI", 0, 1000), ("FEM", 500, 700)]))
    assert _spans(result) == [["CHI", 0, 1000]]


# ============================================================ cva
def test_cva_matches_truth(vtc):
    restrictive = RemoveOverlapsDerivator().derive(None, {}, vtc)
    result = CVADerivator().derive(None, {}, restrictive)
    pd.testing.assert_frame_equal(result, pd.read_csv(DERIVATIONS_TRUTH / "cva.csv", keep_default_na=False), check_dtype=False)


def test_cva_restrictive():
    result = CVADerivator(scenario="R").derive(None, {}, segments([
        ("FEM", 0, 100), ("CHI", 200, 300), ("MAL", 400, 500), ("OCH", 10000, 10100), ("FEM", 10200, 10300),
        ("CHI", 16000, 16100)]))
    assert result["cva"].tolist() == ["Y", "NA", "Y", "N", "N", "NA"]


def test_cva_speaker_class():
    result = CVADerivator().derive(None, {}, segments([("CHI", 0, 100), ("FEM", 200, 300)]))
    assert result["speaker_class"].tolist() == ["C", "O"]


@pytest.mark.parametrize("rows,expected", [
    # direct neighbour of the key child
    ([("FEM", 0, 100), ("CHI", 200, 300), ("MAL", 5000, 5100)], ["Y", "NA", "Y"]),
    # turns between different adults are not child-directed; isolated vocalization defaults to Y
    ([("FEM", 0, 100), ("MAL", 200, 300), ("FEM", 400, 500), ("OCH", 20000, 20100)], ["N", "N", "N", "Y"]),
    # turns with the same speaker type are child-directed (closest neighbour wins)
    ([("FEM", 0, 100), ("FEM", 150, 200), ("FEM", 1000, 1100)], ["Y", "Y", "Y"]),
    ([("MAL", 0, 100), ("FEM", 150, 200), ("FEM", 1000, 1100), ("FEM", 1150, 1200)], ["N", "N", "Y", "Y"]),
])
def test_cva_permissive(rows, expected):
    assert CVADerivator(scenario="P").derive(None, {}, segments(rows))["cva"].tolist() == expected


def test_cva_iti():
    rows = [("FEM", 0, 100), ("CHI", 1100, 1200)]
    assert CVADerivator(iti=500).derive(None, {}, segments(rows))["cva"].tolist() == ["N", "NA"]
    assert CVADerivator(iti=1000).derive(None, {}, segments(rows))["cva"].tolist() == ["Y", "NA"]


def test_cva_with_non_default_index():
    """segments filtered from a bigger dataframe keep their original index"""
    df = segments([("SPEECH", 0, 50), ("FEM", 0, 100), ("CHI", 200, 300)]).iloc[1:]
    result = CVADerivator().derive(None, {}, df)
    assert result.shape[0] == 2
    assert result["cva"].tolist() == ["Y", "NA"]


# ============================================================ acoustics
def test_acoustic_derivator_matches_truth(project, vtc):
    result = AcousticDerivator(profile=None, target_sr=4096).derive(project, {"recording_filename": "sound.wav"}, vtc)
    pd.testing.assert_frame_equal(result, pd.read_csv(DERIVATIONS_TRUTH / "acoustics.csv"), check_dtype=False,
                                  check_exact=False, rtol=1e-4, atol=1e-7)


def test_acoustic_derivator_adds_pitch_columns(project, vtc):
    result = AcousticDerivator(profile=None, target_sr=4096).derive(project, {"recording_filename": "sound.wav"}, vtc)
    for column in ["mean_pitch_semitone", "median_pitch_semitone", "p5_pitch_semitone", "p95_pitch_semitone",
                   "pitch_range_semitone", "mean_abs_amp", "peak_abs_amp"]:
        assert column in result.columns
    assert not any(c.startswith("raw_pitch") or c.startswith("frame_") for c in result.columns)


def test_acoustic_derivator_sampling_rate_mismatch(project, vtc):
    with pytest.raises(AssertionError):
        AcousticDerivator(profile=None, target_sr=16000).derive(project, {"recording_filename": "sound.wav"}, vtc)


def test_f2st():
    assert AcousticDerivator.f2st(50) == pytest.approx(0)
    assert AcousticDerivator.f2st(100) == pytest.approx(12)
    assert AcousticDerivator.f2st(np.array([50, 200])).tolist() == pytest.approx([0, 24])


def test_get_pitch_without_transformation():
    sr = 4096
    signal = np.sin(2 * np.pi * 220 * np.arange(sr) / sr)
    pitch = AcousticDerivator.get_pitch(signal, sr)
    assert pitch["mean_pitch_f0"] == pytest.approx(220, rel=0.02)
    assert pitch["peak_abs_amp"] == pytest.approx(1, rel=0.01)
