"""AnnotationManager.derive_annotations and AnnotationManager.merge_sets / merge_annotations"""
import datetime
import logging

import pandas as pd
import pytest
import yaml

from ChildProject import __version__
from ChildProject.annotations import ANNOTATIONS, CONVERTED, METANNOTS, AnnotationManager
from ChildProject.pipelines.derivations import AcousticDerivator, Derivator
from ChildProject.projects import EXTRA, METADATA_FOLDER

from conftest import EXAMPLES, import_static

INPUT = EXAMPLES / "valid_raw_data" / "annotations" / "input.csv"
TODAY = datetime.date.today().strftime("%Y-%m-%d")


class CustomDerivator(Derivator):
    def __init__(self, kind="normal"):
        self.kind = kind
        super().__init__()

    def derive(self, project, metadata, segments):
        if self.kind == "number":
            return 1
        if self.kind == "columns":
            return pd.DataFrame([], columns=["segment_onset", "segment_offset"])
        if self.kind == "raise":
            raise RuntimeError("derivation failed")
        if self.kind == "metadata":
            return segments.assign(child_dob=metadata["child_dob"], set_method=metadata["method"])
        return segments

    def get_auto_metadata(self, am, input_set, output_set):
        return {"has_speaker_type": "Y", "segmentation": output_set}


def converted_files(am, set_name):
    return sorted(p.name for p in (am.project.path / ANNOTATIONS / set_name / CONVERTED).glob("*.csv"))


def lines(am, set_name):
    return am.annotations[am.annotations["set"] == set_name]


# ============================================================ derive_annotations
def test_derive_creates_one_line_per_input_line(am):
    imported, errors = am.derive_annotations("vtc_present", "output", CustomDerivator(), threads=1)
    assert imported.shape[0] == 2
    assert errors is None or errors.shape[0] == 0
    assert lines(am, "output").shape[0] == 2


def test_derive_index_lines(am):
    imported, _ = am.derive_annotations("vtc_present", "output", CustomDerivator(), threads=1)
    expected = lines(am, "vtc_present").assign(merged_from="vtc_present", set="output", format="NA")
    columns = ["set", "recording_filename", "time_seek", "range_onset", "range_offset", "raw_filename", "format",
               "annotation_filename", "merged_from"]
    pd.testing.assert_frame_equal(expected[columns].reset_index(drop=True), imported[columns].reset_index(drop=True),
                                  check_dtype=False)
    assert (imported["package_version"] == __version__).all()


def test_derive_writes_converted_files(am):
    am.derive_annotations("vtc_present", "output", CustomDerivator(), threads=1)
    assert converted_files(am, "output") == ["sound2_0_4000.csv", "sound_0_4000.csv"]
    pd.testing.assert_frame_equal(
        pd.read_csv(am.project.path / ANNOTATIONS / "output" / CONVERTED / "sound_0_4000.csv"),
        pd.read_csv(am.project.path / ANNOTATIONS / "vtc_present" / CONVERTED / "sound_0_4000.csv").sort_values(
            ["segment_onset", "segment_offset", "speaker_type"]).reset_index(drop=True),
    )


def test_derive_writes_index(am):
    am.derive_annotations("vtc_present", "output", CustomDerivator(), threads=1)
    on_disk = pd.read_csv(am.project.path / METADATA_FOLDER / "annotations.csv")
    assert (on_disk["set"] == "output").sum() == 2


def test_derive_set_metadata(am):
    am.derive_annotations("vtc_present", "output", CustomDerivator(), threads=1,
                          derivation_metadata={"segmentation_type": "restrictive", "has_words": "Y"})
    meta = yaml.safe_load((am.project.path / ANNOTATIONS / "output" / METANNOTS).read_text())
    assert meta.pop("derivator_object").startswith("<test_annotations_derive_merge.CustomDerivator")
    assert meta == {"segmentation": "output", "segmentation_type": "restrictive", "has_words": "Y",
                    "has_speaker_type": "Y", "method": "derivation", "date_annotation": TODAY,
                    "ChildProject_version": __version__}


def test_derive_metadata_is_available_after_derivation(am):
    am.derive_annotations("vtc_present", "output", CustomDerivator(), threads=1)
    meta = am.get_sets_metadata().loc["output"]
    assert meta["method"] == "derivation" and meta["duration"] == 8000


def test_derive_passes_merged_metadata_to_derivator(am):
    am.derive_annotations("vtc_present", "output", CustomDerivator("metadata"), threads=1)
    derived = pd.read_csv(am.project.path / ANNOTATIONS / "output" / CONVERTED / "sound_0_4000.csv")
    assert derived["child_dob"].unique().tolist() == ["2020-01-01"]   # children metadata
    assert derived["set_method"].unique().tolist() == ["automated"]   # set metadata


def test_derive_with_builtin_derivation_name(am):
    am.derive_annotations("vtc_present", "conv", "conversations", threads=1)
    derived = pd.read_csv(am.project.path / ANNOTATIONS / "conv" / CONVERTED / "sound_0_4000.csv")
    assert {"is_CT", "conv_count"} <= set(derived.columns)
    assert yaml.safe_load((am.project.path / ANNOTATIONS / "conv" / METANNOTS).read_text())["has_interactions"] == "Y"


def test_derive_with_function(am):
    imported, _ = am.derive_annotations("vtc_present", "output", lambda project, metadata, segments: segments, threads=1)
    assert imported.shape[0] == 2


def test_derive_with_pool(am):
    imported, _ = am.derive_annotations("vtc_present", "output", CustomDerivator(), threads=2)
    assert imported.shape[0] == 2


def test_derive_output_as_path(am, tmp_path):
    imported, _ = am._derive_annotations("vtc_present", str(tmp_path / "out"), CustomDerivator(), threads=1,
                                         output_as_path=True)
    assert imported.shape[0] == 2
    assert sorted(p.name for p in (tmp_path / "out" / CONVERTED).glob("*.csv")) == ["sound2_0_4000.csv", "sound_0_4000.csv"]
    assert (tmp_path / "out" / METANNOTS).exists()
    assert pd.read_csv(am.project.path / METADATA_FOLDER / "annotations.csv").shape[0] == 2  # index untouched


@pytest.mark.parametrize("input_set,output_set,derivation,error", [
    ("missing", "output", CustomDerivator(), AssertionError),
    ("vtc_present", "vtc_present", CustomDerivator(), AssertionError),
    ("vtc_present", "output", "not_a_derivation", ValueError),
])
def test_derive_rejects_invalid_arguments(am, input_set, output_set, derivation, error):
    with pytest.raises(error):
        am.derive_annotations(input_set, output_set, derivation, threads=1)


@pytest.mark.parametrize("derivation", [
    CustomDerivator("number"),                          # does not return a dataframe
    CustomDerivator("columns"),                         # misses required columns
    CustomDerivator("raise"),                           # raises
    lambda a, b: b,                                     # wrong signature
])
def test_derive_failures_are_reported_per_line(am, derivation):
    imported, errors = am.derive_annotations("vtc_present", "output", derivation, threads=1)
    assert imported.shape[0] == 0
    assert errors.shape[0] == 2
    assert errors["error"].notnull().all()


def test_derive_failures_do_not_write_metadata(am):
    am.derive_annotations("vtc_present", "output", CustomDerivator("number"), threads=1)
    assert not (am.project.path / ANNOTATIONS / "output" / METANNOTS).exists()


def test_derive_errors_are_exported_to_extra(am):
    am.derive_annotations("vtc_present", "output", CustomDerivator("raise"), threads=1)
    assert len(list((am.project.path / EXTRA).glob("errors_derive_*.csv"))) == 1


@pytest.fixture
def existing_output(am):
    am.derive_annotations("vtc_present", "output", CustomDerivator(), threads=1)
    return am


def test_derive_existing_output_without_overwrite_keeps_files(existing_output, caplog):
    path = existing_output.project.path / ANNOTATIONS / "output" / CONVERTED / "sound_0_4000.csv"
    path.write_text("segment_onset,segment_offset\n")
    with caplog.at_level(logging.WARNING, logger="ChildProject.annotations"):
        existing_output.derive_annotations("vtc_present", "output", CustomDerivator(), threads=1)
    assert path.read_text() == "segment_onset,segment_offset\n"
    assert "already exists" in caplog.text


def test_derive_existing_output_with_overwrite(existing_output, caplog):
    path = existing_output.project.path / ANNOTATIONS / "output" / CONVERTED / "sound_0_4000.csv"
    path.write_text("segment_onset,segment_offset\n")
    with caplog.at_level(logging.WARNING, logger="ChildProject.annotations"):
        existing_output.derive_annotations("vtc_present", "output", CustomDerivator(), threads=1, overwrite_existing=True)
    assert pd.read_csv(path).shape[0] > 0
    assert "will be overwritten" in caplog.text
    assert lines(existing_output, "output").shape[0] == 2


def test_derive_overlapping_existing_output(am, caplog):
    other = am.annotations.iloc[[0]].assign(set="output", range_offset=3000, annotation_filename="other.csv")
    am.annotations = pd.concat([am.annotations, other])
    with caplog.at_level(logging.ERROR, logger="ChildProject.annotations"):
        imported, errors = am.derive_annotations("vtc_present", "output", CustomDerivator(), threads=1)
    assert errors.shape[0] == 1
    assert "cannot continue because it overlaps with these existing annotation lines" in errors.iloc[0]["error"]
    assert "cannot continue because it overlaps" in caplog.text


def test_derive_logs_when_set_metadata_cannot_be_written(am, monkeypatch, caplog):
    def fail(*args, **kwargs):
        raise OSError("read-only")

    monkeypatch.setattr(am, "_write_set_metadata", fail)
    with caplog.at_level(logging.ERROR, logger="ChildProject.annotations"):
        imported, _ = am.derive_annotations("vtc_present", "output", CustomDerivator(), threads=1)
    assert imported.shape[0] == 2
    assert "Could not write set metadata for output" in caplog.text


def test_derive_acoustics_skips_video_recordings(am):
    """AcousticDerivator declares MEDIA_TYPES=['audio']: video recordings must be skipped, not fail"""
    segments = pd.read_csv(am.project.path / ANNOTATIONS / "vtc_present" / CONVERTED / "sound_0_4000.csv")
    import_static(am, segments, "vtc_present", recording_filename="video.mp4", range_offset=4000)
    imported, errors = am.derive_annotations("vtc_present", "acoustics", AcousticDerivator(profile=None, target_sr=4096),
                                             threads=1)
    assert sorted(imported["recording_filename"]) == ["sound.wav", "sound2.wav"]
    assert errors is None or errors.shape[0] == 0


# ============================================================ merge_sets
@pytest.fixture
def mergeable(long_am):
    input_annotations = pd.read_csv(INPUT)
    long_am.import_annotations(input_annotations[input_annotations["set"].isin(["vtc_rttm", "alice/output"])], threads=1)
    return long_am


MERGE = dict(left_set="vtc_rttm", right_set="alice/output", left_columns=["speaker_type"],
             right_columns=["phonemes", "syllables", "words"], output_set="alice_vtc", threads=1)


def test_merge_sets_creates_one_line_per_intersection(mergeable):
    assert mergeable.merge_sets(**MERGE) is mergeable
    merged = lines(mergeable, "alice_vtc")
    assert sorted(merged["recording_filename"]) == ["sound.wav", "sound2.wav"]
    assert (merged["merged_from"] == "vtc_rttm,alice/output").all()
    assert (merged["format"] == "NA").all() and (merged["time_seek"] == 0).all()


def test_merge_sets_writes_index(mergeable):
    mergeable.merge_sets(**MERGE)
    on_disk = pd.read_csv(mergeable.project.path / METADATA_FOLDER / "annotations.csv")
    assert (on_disk["set"] == "alice_vtc").sum() == 2


def test_merge_sets_segments(mergeable):
    mergeable.merge_sets(**MERGE)
    merged = mergeable.get_segments(lines(mergeable, "alice_vtc"))
    vtc = mergeable.get_segments(lines(mergeable, "vtc_rttm"))
    assert merged.shape[0] == vtc.shape[0]
    assert merged.shape[1] == vtc.shape[1] + 3


def test_merge_sets_copies_right_columns_on_matching_segments(mergeable):
    mergeable.merge_sets(**MERGE)
    merged = mergeable.get_segments(lines(mergeable, "alice_vtc"))
    adults = merged[merged["speaker_type"].isin(["FEM", "MAL"])].sort_values(["segment_onset", "segment_offset"])
    alice = mergeable.get_segments(lines(mergeable, "alice/output")).sort_values(["segment_onset", "segment_offset"])
    pd.testing.assert_frame_equal(adults[["phonemes", "syllables", "words"]].reset_index(drop=True),
                                  alice[["phonemes", "syllables", "words"]].reset_index(drop=True))


def test_merge_sets_raw_filename_lists_both_sources(mergeable):
    mergeable.merge_sets(**MERGE)
    assert set(lines(mergeable, "alice_vtc")["raw_filename"]) == {"example.rttm,example_alice.txt"}


def test_merge_sets_recording_filter(mergeable):
    mergeable.merge_sets(**dict(MERGE, recording_filter={"sound.wav"}))
    assert lines(mergeable, "alice_vtc")["recording_filename"].tolist() == ["sound.wav"]


def test_merge_sets_skip_existing(mergeable):
    mergeable.merge_sets(**dict(MERGE, recording_filter={"sound.wav"}))
    first = mergeable.project.path / ANNOTATIONS / "alice_vtc" / CONVERTED / "sound_1980000_1990000.csv"
    first.write_text("segment_onset,segment_offset\n")
    mergeable.merge_sets(**dict(MERGE, full_set_merge=False, skip_existing=True))
    assert sorted(lines(mergeable, "alice_vtc")["recording_filename"]) == ["sound.wav", "sound2.wav"]
    assert first.read_text() == "segment_onset,segment_offset\n"


def test_merge_sets_extra_columns(mergeable):
    """documented as 'additional columns to add to the resulting converted annotations'"""
    mergeable.merge_sets(**dict(MERGE, columns={"speaker_id": "merged"}))
    converted = pd.read_csv(mergeable.project.path / ANNOTATIONS / "alice_vtc" / CONVERTED / "sound_1980000_1990000.csv")
    assert "speaker_id" in converted.columns
    assert set(converted["speaker_id"]) == {"merged"}


@pytest.mark.parametrize("metadata,source", [("left", "vtc_rttm"), ("right", "alice/output")])
def test_merge_sets_inherits_metadata(mergeable, metadata, source):
    (mergeable.project.path / ANNOTATIONS / "vtc_rttm" / METANNOTS).write_text("method: automated\nannotation_algorithm_name: VTC\n")
    mergeable._read_sets_metadata()
    mergeable.merge_sets(**dict(MERGE, metadata=metadata))
    meta = mergeable.get_sets_metadata().loc["alice_vtc"]
    expected = mergeable.get_sets_metadata().loc[source]
    assert meta["method"] == expected["method"]
    assert meta["date_annotation"] == TODAY
    assert (meta["has_speaker_type"], meta["has_words"]) == ("Y", "Y")


def test_merge_sets_default_metadata(mergeable):
    mergeable.merge_sets(**MERGE)
    meta = yaml.safe_load((mergeable.project.path / ANNOTATIONS / "alice_vtc" / METANNOTS).read_text())
    assert meta == {"date_annotation": TODAY, "has_speaker_type": "Y", "has_words": "Y"}


def test_merge_sets_keeps_existing_metadata_file(mergeable):
    (mergeable.project.path / ANNOTATIONS / "alice_vtc").mkdir()
    (mergeable.project.path / ANNOTATIONS / "alice_vtc" / METANNOTS).write_text("method: manual\n")
    mergeable.merge_sets(**MERGE)
    assert (mergeable.project.path / ANNOTATIONS / "alice_vtc" / METANNOTS).read_text() == "method: manual\n"


def _fail_writing(*args, **kwargs):
    """module-level so that the patched manager can still be pickled by merge_sets' pool"""
    raise OSError("read-only")


def test_merge_sets_logs_when_set_metadata_cannot_be_written(mergeable, monkeypatch, caplog):
    monkeypatch.setattr(mergeable, "_write_set_metadata", _fail_writing)
    with caplog.at_level(logging.ERROR, logger="ChildProject.annotations"):
        mergeable.merge_sets(**MERGE)
    assert "Could not write set metadata for alice_vtc" in caplog.text


@pytest.mark.parametrize("overrides,message", [
    ({"output_set": "vtc_rttm"}, "output_set <vtc_rttm> already exists"),
    ({"left_set": "nope"}, "left_set <nope> was not found"),
    ({"right_set": "nope"}, "right_set <nope> was not found"),
    ({"metadata": "both"}, "metadata parameter unrecognized"),
    ({"right_set": "vtc_rttm"}, "sets must differ"),
    ({"right_columns": ["speaker_type"]}, "left_columns and right_columns must be disjoint"),
    ({"right_columns": ["not_a_column"]}, "left_columns and right_columns have unexpected values"),
])
def test_merge_sets_rejects_invalid_arguments(mergeable, overrides, message):
    with pytest.raises(AssertionError, match=message):
        mergeable.merge_sets(**dict(MERGE, **overrides))


def test_merge_sets_without_intersection(mergeable):
    with pytest.raises(ValueError, match="No intersection was found"):
        mergeable.merge_sets(**dict(MERGE, recording_filter={"video.mp4"}))


def test_merge_sets_ignores_lines_in_error(mergeable):
    mergeable.annotations["error"] = mergeable.annotations["error"].astype(object)
    mergeable.annotations.loc[mergeable.annotations["set"] == "alice/output", "error"] = "broken"
    with pytest.raises(ValueError, match="No intersection was found"):
        mergeable.merge_sets(**MERGE)


def _merge_input(am, recording):
    intersection = AnnotationManager.intersection(am.annotations, sets=["vtc_rttm", "alice/output"])
    sides = {}
    for side, set_name in [("left_annotations", "vtc_rttm"), ("right_annotations", "alice/output")]:
        df = intersection[intersection["set"] == set_name].reset_index(drop=True).rename_axis("interval").reset_index()
        sides[side] = df[df["recording_filename"] == recording]
    return sides


def test_merge_annotations_direct_call(mergeable):
    """merge_sets runs merge_annotations in a pool; call it directly to check one recording's result"""
    result = mergeable.merge_annotations(["speaker_type"], ["words"], {}, "direct", _merge_input(mergeable, "sound.wav"))
    assert result["annotation_filename"].tolist() == ["sound_1980000_1990000.csv"]
    assert result["set"].tolist() == ["direct"]
    converted = pd.read_csv(mergeable.project.path / ANNOTATIONS / "direct" / CONVERTED / "sound_1980000_1990000.csv")
    assert {"segment_onset", "segment_offset", "speaker_type", "words", "raw_filename"} == set(converted.columns)


def test_merge_annotations_skip_existing_direct_call(mergeable):
    (mergeable.project.path / ANNOTATIONS / "direct" / CONVERTED).mkdir(parents=True)
    (mergeable.project.path / ANNOTATIONS / "direct" / CONVERTED / "sound_1980000_1990000.csv").write_text("x\n")
    result = mergeable.merge_annotations(["speaker_type"], ["words"], {}, "direct",
                                         _merge_input(mergeable, "sound.wav"), skip_existing=True)
    assert result.shape[0] == 0


def test_merge_annotations_missing_converted_file(mergeable):
    (mergeable.project.path / ANNOTATIONS / "vtc_rttm" / CONVERTED / "sound_1980000_1990000.csv").unlink()
    with pytest.raises(Exception, match="the following annotations from the left set are missing"):
        mergeable.merge_sets(**MERGE)


def test_merge_annotations_missing_right_converted_file(mergeable):
    (mergeable.project.path / ANNOTATIONS / "alice/output" / CONVERTED / "sound_1980000_1990000.csv").unlink()
    with pytest.raises(Exception, match="the following annotations from the right set are missing"):
        mergeable.merge_sets(**MERGE)
