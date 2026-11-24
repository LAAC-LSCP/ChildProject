"""AnnotationManager: index reading/writing, sets metadata, file management, set operations and queries.
Importation is tested in test_annotations_import.py, derivation and merging in test_annotations_derive_merge.py"""
import datetime
import filecmp
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ChildProject.annotations import ANNOTATIONS, CONVERTED, METANNOTS, RAW, AnnotationManager
from ChildProject.projects import METADATA_FOLDER, ChildProject
from ChildProject.tables import MissingColumnsException
from ChildProject.utils import TimeInterval

from conftest import EXAMPLES, TEST_DATA, TRUTH

INPUT = EXAMPLES / "valid_raw_data" / "annotations" / "input.csv"
SETS_DTYPES = {f.name: f.dtype if f.dtype is not None else "string" for f in AnnotationManager.SETS_COLUMNS}
OLD_ITS_MISSING = (
    "Metadata file content for sets ['old_its'] could not be found, it may be downloaded from a remote with the "
    f"command `datalad get {Path('annotations/**/metannots.yml')}`"
)
UNKNOWN_FIELDS = (
    "Metadata files for sets contain the following unknown fields ['invented', 'random_field'] which can be found in "
    "the metadata for sets ['alice/output', 'textgrid']"
)


def standardize(df, columns):
    df = df[list(columns)]
    return df.sort_index(axis=1).sort_values(list(columns)).reset_index(drop=True)


def index_rows(am, rows):
    """replace the annotation index by ``rows`` (dicts) and write it"""
    am.annotations = pd.DataFrame(rows)
    return am


def write_index(project, rows):
    pd.DataFrame(rows).to_csv(project.path / METADATA_FOLDER / "annotations.csv", index=False)


VTC_PRESENT = {"set": "vtc_present", "recording_filename": "sound.wav", "time_seek": 0, "range_onset": 0,
               "range_offset": 4000, "raw_filename": "example.rttm", "format": "vtc_rttm",
               "annotation_filename": "sound_0_4000.csv", "imported_at": "2024-03-12 12:58:44",
               "package_version": "0.1.1"}


# ============================================================ constructor
def test_init_requires_a_childproject():
    with pytest.raises(ValueError, match="project should derive from ChildProject"):
        AnnotationManager("examples/valid_raw_data")


def test_init_reads_the_index(am):
    assert am.annotations["annotation_filename"].tolist() == ["sound2_0_4000.csv", "sound_0_4000.csv"]


def test_init_reads_the_project_if_needed(valid_path):
    project = ChildProject(valid_path)
    AnnotationManager(project)
    assert project.recordings is not None


def test_init_creates_missing_index(valid_path):
    (valid_path / METADATA_FOLDER / "annotations.csv").unlink()
    am = AnnotationManager(ChildProject(valid_path))
    assert am.annotations.shape[0] == 0
    header = (valid_path / METADATA_FOLDER / "annotations.csv").read_text().split(",")
    assert header == [c.name for c in AnnotationManager.INDEX_COLUMNS]


def test_init_stores_read_errors_and_warnings(am):
    assert am.errors == [] and am.warnings == []


# ============================================================ read
def test_read_returns_errors_and_warnings(am):
    assert am.read() == ([], [])


def test_read_duplicate_annotation_files(am):
    write_index(am.project, [VTC_PRESENT, VTC_PRESENT])
    errors, _ = am.read()
    assert errors == ["duplicate reference to annotations/vtc_present/converted/sound_0_4000.csv (appears 2 times)"]


@pytest.mark.parametrize("onset,offset", [(100, 100), (200, 100), (-1, 100)])
def test_read_invalid_ranges(am, onset, offset):
    write_index(am.project, [dict(VTC_PRESENT, range_onset=onset, range_offset=offset)])
    errors, _ = am.read()
    assert "annotation index does not verify range_offset > range_onset >= 0 for set <vtc_present>, annotation filename <sound_0_4000.csv>" in errors


def test_read_range_beyond_recording_duration(am):
    write_index(am.project, [dict(VTC_PRESENT, range_offset=4001)])
    errors, _ = am.read()
    assert errors == ["annotation index has an offset higher than recorded duration of the audio <vtc_present>, annotation filename <sound_0_4000.csv>"]


def test_read_without_duration_does_not_check_ranges(am):
    write_index(am.project, [dict(VTC_PRESENT, range_offset=99999)])
    am.project.recordings = am.project.recordings.drop(columns=["duration"])
    assert am.read()[0] == []


def test_read_warns_about_outdated_merged_sets(am):
    derived = dict(VTC_PRESENT, set="derived", merged_from="vtc_present", imported_at="2020-01-01 00:00:00")
    write_index(am.project, [VTC_PRESENT, derived])
    _, warnings = am.read()
    assert any("set derived is outdated because the vtc_present set" in w for w in warnings)


# ============================================================ sets metadata
@pytest.fixture
def all_sets(am):
    """fake index with one line per annotation set folder, so that every metannots.yml is read"""
    sets = [n if n != "alice" else "alice/output" for n in os.listdir(am.project.path / ANNOTATIONS)
            if os.path.isdir(am.project.path / ANNOTATIONS / n)]
    am.annotations = pd.DataFrame({"set": sets, "range_onset": 0, "range_offset": 0,
                                   "annotation_filename": "", "raw_filename": ""})
    return am


@pytest.fixture
def no_metadata(all_sets):
    for s in all_sets.annotations["set"].unique():
        path = all_sets.project.path / ANNOTATIONS / s / METANNOTS
        if path.exists():
            path.unlink()
    return all_sets


def read_truth_sets(name):
    return pd.read_csv(TRUTH / name, index_col="set", dtype=SETS_DTYPES, encoding="utf-8")


def test_read_sets_metadata(all_sets):
    result = all_sets._read_sets_metadata()
    pd.testing.assert_frame_equal(read_truth_sets("sets_metadata.csv").drop(columns="duration"), result,
                                  check_like=True, check_dtype=False)


def test_read_sets_metadata_stores_sets(all_sets):
    result = all_sets._read_sets_metadata()
    assert all_sets.sets is result


def test_read_sets_metadata_return_warnings(all_sets):
    _, warnings = all_sets._read_sets_metadata("return")
    assert warnings == [
        "Metadata files for sets ['vtc_rttm'] could not be found, they should be created as annotations/<set>/metannots.yml",
        OLD_ITS_MISSING,
        UNKNOWN_FIELDS,
    ]


def test_read_sets_metadata_log_warnings(all_sets, caplog):
    with caplog.at_level(logging.WARNING, logger="ChildProject.annotations"):
        all_sets._read_sets_metadata("log")
    assert [r.message for r in caplog.records] == [
        "Metadata files for sets ['vtc_rttm'] could not be found, they should be created as annotations/<set>/metannots.yml",
        OLD_ITS_MISSING,
        UNKNOWN_FIELDS,
    ]


def test_read_sets_metadata_ignore_is_silent(all_sets, caplog):
    with caplog.at_level(logging.WARNING, logger="ChildProject.annotations"):
        all_sets._read_sets_metadata("ignore")
    assert caplog.records == []


def test_read_sets_metadata_without_metadata_files(no_metadata):
    result, warnings = no_metadata._read_sets_metadata("return")
    pd.testing.assert_frame_equal(read_truth_sets("sets_empty_metadata.csv").drop(columns="duration"), result,
                                  check_like=True, check_dtype=False)
    assert warnings == [
        "Metadata files for sets ['alice/output', 'eaf_basic', 'eaf_solis', 'metrics', 'new_its', 'textgrid', "
        "'textgrid2', 'vtc_present', 'vtc_rttm'] could not be found, they should be created as annotations/<set>/metannots.yml",
        OLD_ITS_MISSING,
    ]


def test_read_sets_metadata_invalid_warning_mode(am):
    with pytest.raises(ValueError, match="warning argument must be in"):
        am._read_sets_metadata("shout")


def test_read_sets_metadata_wrong_types(am):
    path = am.project.path / ANNOTATIONS / "vtc_present" / METANNOTS
    path.write_text("sampling_count: many\n")
    _, warnings = am._read_sets_metadata("return")
    assert any(w.startswith("Could not convert metadata to expected types") for w in warnings)


def test_get_sets_metadata(all_sets):
    pd.testing.assert_frame_equal(read_truth_sets("sets_metadata.csv"), all_sets.get_sets_metadata(),
                                  check_like=True, check_dtype=False)


def test_get_sets_metadata_without_metadata_files(no_metadata):
    pd.testing.assert_frame_equal(read_truth_sets("sets_empty_metadata.csv"), no_metadata.get_sets_metadata(),
                                  check_like=True, check_dtype=False)


def test_get_sets_metadata_duration_is_sum_of_ranges(am):
    assert am.get_sets_metadata().loc["vtc_present", "duration"] == 8000


def test_get_sets_metadata_csv(am):
    csv = am.get_sets_metadata("csv")
    assert csv.splitlines()[0].startswith("set,")
    assert csv.splitlines()[1].startswith("vtc_present,")


def test_get_sets_metadata_csv_custom_delimiter(am):
    assert am.get_sets_metadata("csv", delimiter=";").splitlines()[0].startswith("set;")


def test_get_sets_metadata_lslike(am):
    assert am.get_sets_metadata("lslike") == (
        "duration    method algo version       date transcr\n"
        "    8000 automated  VTC       1 2024-04-07         \x1b[94mvtc_present\x1b[0m\n"
    )


def test_get_sets_metadata_lslike_human_readable(am):
    assert "    8.0s automated" in am.get_sets_metadata("lslike", human=True)


def test_get_sets_metadata_sorting(all_sets):
    names = all_sets.get_sets_metadata(sort_by="set", sort_ascending=False).index.tolist()
    assert names == sorted(names, reverse=True)


def test_get_sets_metadata_unknown_format(am):
    with pytest.raises(ValueError, match="format <xml> is unknown"):
        am.get_sets_metadata("xml")


SETS_TRUTH = pd.read_csv(TRUTH / "sets_metadata.csv", index_col="set")


@pytest.mark.parametrize("sets,delimiter,header,human,truth_file", [
    (SETS_TRUTH, " ", True, False, "printable_meta_default.txt"),
    (SETS_TRUTH, "=", True, False, "printable_meta_eq.txt"),
    (SETS_TRUTH, " ", False, False, "printable_meta_no-header.txt"),
    (SETS_TRUTH, " ", True, True, "printable_meta_hr.txt"),
    (SETS_TRUTH, "=", False, False, "printable_meta_eq_no-header.txt"),
    (SETS_TRUTH, " ", False, True, "printable_meta_hr_no-header.txt"),
    (SETS_TRUTH, "=", False, True, "printable_meta_eq_hr_no-header.txt"),
    (SETS_TRUTH, "=", True, True, "printable_meta_eq_hr.txt"),
    (pd.DataFrame(), " ", True, False, "printable_meta_empty.txt"),
])
def test_get_printable_sets_metadata(sets, delimiter, header, human, truth_file):
    expected = (TRUTH / "printable" / truth_file).read_text().encode("utf-8").decode("unicode_escape")
    assert AnnotationManager.get_printable_sets_metadata(sets, delimiter, header, human) == expected


def test_get_printable_sets_metadata_invalid_delimiter():
    with pytest.raises(ValueError):
        AnnotationManager.get_printable_sets_metadata(SETS_TRUTH, "xx", True, True)


def test_get_printable_sets_metadata_requires_dataframe():
    with pytest.raises(AssertionError):
        AnnotationManager.get_printable_sets_metadata("notadataframe", " ")


def test_write_set_metadata(am):
    am._write_set_metadata("vtc_present", {"method": "manual"})
    assert yaml.safe_load((am.project.path / ANNOTATIONS / "vtc_present" / METANNOTS).read_text()) == {"method": "manual"}


def test_write_set_metadata_unknown_set(am):
    with pytest.raises(AssertionError, match="set must exist"):
        am._write_set_metadata("nope", {"method": "manual"})


def test_write_set_metadata_to_path(am, tmp_path):
    am._write_set_metadata(tmp_path, {"method": "manual"}, output_as_path=True)
    assert yaml.safe_load((tmp_path / METANNOTS).read_text()) == {"method": "manual"}


# ============================================================ outdated merged sets
@pytest.fixture
def merged_index(am):
    am.annotations = pd.DataFrame([
        {"set": "a", "imported_at": "2024-01-02 00:00:00", "merged_from": np.nan},
        {"set": "b", "imported_at": "2024-01-01 00:00:00", "merged_from": np.nan},
        {"set": "ab", "imported_at": "2024-01-01 12:00:00", "merged_from": "a,b"},
        {"set": "ghost", "imported_at": "2024-01-03 00:00:00", "merged_from": "gone"},
    ])
    return am


def test_check_outdated_merged_sets(merged_index):
    warnings = merged_index._check_for_outdated_merged_sets()
    assert len(warnings) == 2
    assert warnings[0].startswith("set ab is outdated because the a set it is merged from was modified.")
    assert warnings[1].startswith("set ghost was originally derived from set gone which does not exist anymore.")


def test_check_outdated_merged_sets_restricted_to_sets(merged_index):
    assert merged_index._check_for_outdated_merged_sets(sets={"b"}) == []
    assert len(merged_index._check_for_outdated_merged_sets(sets={"a"})) == 1


def test_check_outdated_merged_sets_without_merged_from(am):
    assert am._check_for_outdated_merged_sets() == []


# ============================================================ annotation files
SOURCE = "metadata/children/0_test.csv"


@pytest.mark.parametrize("dst_file,set_name,dst_path", [
    ("rec1.rttm", "new_vtc", "annotations/new_vtc/raw/rec1.rttm"),
    (Path("rec1.rttm"), "new_vtc", "annotations/new_vtc/raw/rec1.rttm"),
    ("example2.rttm", "vtc_rttm", "annotations/vtc_rttm/raw/example2.rttm"),
    ("scripts/new_subfolder/any_script.py", "super/long/set", "annotations/super/long/set/raw/scripts/new_subfolder/any_script.py"),
])
def test_add_annotation_file(am, dst_file, set_name, dst_path):
    src = am.project.path / SOURCE
    assert am.add_annotation_file(src, dst_file, set_name, False) is am
    assert filecmp.cmp(src, am.project.path / dst_path, shallow=False)


def test_add_annotation_file_str_source(am):
    am.add_annotation_file(str(am.project.path / SOURCE), "rec1.rttm", "new_vtc", False)
    assert (am.project.path / "annotations/new_vtc/raw/rec1.rttm").exists()


def test_add_annotation_file_existing(am):
    with pytest.raises(AssertionError, match="already exists"):
        am.add_annotation_file(am.project.path / SOURCE, "example.rttm", "vtc_rttm", False)


def test_add_annotation_file_overwrite(am):
    am.add_annotation_file(am.project.path / SOURCE, "example.rttm", "vtc_rttm", True)
    assert filecmp.cmp(am.project.path / SOURCE, am.project.path / "annotations/vtc_rttm/raw/example.rttm", shallow=False)


def test_add_annotation_file_missing_source(am):
    with pytest.raises(FileNotFoundError):
        am.add_annotation_file(am.project.path / "made_up.txt", "rec.rttm", "new_vtc", False)


@pytest.mark.parametrize("dst_file", ["/etc/rec1.rttm", "../test.rttm"])
def test_add_annotation_file_outside_set(am, dst_file):
    with pytest.raises(AssertionError):
        am.add_annotation_file(am.project.path / SOURCE, dst_file, "new_vtc", False)


def annexed_link(project, relative):
    """symlink like the ones git-annex/datalad create: pointing into .git/annex/objects of the dataset"""
    target = project.path / ".git" / "annex" / "objects" / "content"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch()
    (project.path / relative).symlink_to(target)


def test_add_annotation_file_refuses_symlink(am):
    annexed_link(am.project, "annotations/vtc_rttm/raw/link.rttm")
    with pytest.raises(AssertionError, match="annexed data"):
        am.add_annotation_file(am.project.path / SOURCE, "link.rttm", "vtc_rttm", True)


def test_add_annotation_file_warns_on_extension_change(am, caplog):
    with caplog.at_level(logging.WARNING, logger="ChildProject.annotations"):
        am.add_annotation_file(am.project.path / SOURCE, "rec1.rttm", "new_vtc", False)
    assert "have different file extensions" in caplog.text


def test_remove_annotation_file(am):
    assert am.remove_annotation_file("example.rttm", "vtc_rttm") is am
    assert not (am.project.path / "annotations/vtc_rttm/raw/example.rttm").exists()


def test_remove_annotation_file_missing(am):
    with pytest.raises(FileNotFoundError):
        am.remove_annotation_file("example2.rttm", "vtc_rttm")


@pytest.mark.parametrize("file", ["/sound.rttm", "../../../../sound.wav"])
def test_remove_annotation_file_outside_set(am, file):
    with pytest.raises(AssertionError):
        am.remove_annotation_file(file, "vtc")


def test_remove_annotation_file_refuses_symlink(am):
    annexed_link(am.project, "annotations/vtc_rttm/raw/link.rttm")
    with pytest.raises(AssertionError, match="annexed data"):
        am.remove_annotation_file("link.rttm", "vtc_rttm")


# ============================================================ rename_recording_filename
def test_rename_recording_filename(am):
    assert am.rename_recording_filename("sound.wav", "new_sound.wav") == "new_sound.wav"
    assert am.annotations["recording_filename"].tolist().count("new_sound.wav") == 1
    assert "sound.wav" not in am.annotations["recording_filename"].tolist()


def test_rename_recording_filename_writes_index(am):
    am.rename_recording_filename("sound.wav", "new_sound.wav")
    on_disk = pd.read_csv(am.project.path / METADATA_FOLDER / "annotations.csv")
    assert "new_sound.wav" in on_disk["recording_filename"].tolist()


def test_rename_recording_filename_not_referenced(am):
    before = am.annotations.copy()
    am.rename_recording_filename("soundx.wav", "new_sound.wav")
    assert am.annotations["recording_filename"].tolist() == before["recording_filename"].tolist()


def test_rename_recording_filename_creating_overlaps(am):
    with pytest.raises(ValueError, match="would cause overlaps"):
        am.rename_recording_filename("sound.wav", "sound2.wav")


# ============================================================ validate
def test_validate_annotation(am):
    assert am.validate_annotation(VTC_PRESENT) == ([], [])


def test_validate_annotation_unreadable_file(am):
    errors, warnings = am.validate_annotation(dict(VTC_PRESENT, annotation_filename="missing.csv"))
    assert errors[0].startswith("error while trying to read missing.csv from vtc_present")
    assert warnings == []


def test_validate_annotation_invalid_segments(am):
    path = am.project.path / ANNOTATIONS / "vtc_present" / CONVERTED / "sound_0_4000.csv"
    df = pd.read_csv(path)
    df.loc[0, "speaker_type"] = "DOG"
    df.to_csv(path, index=False)
    errors, _ = am.validate_annotation(VTC_PRESENT)
    assert len(errors) == 1 and "speaker_type" in errors[0]


def test_validate_annotation_unknown_column(am):
    path = am.project.path / ANNOTATIONS / "vtc_present" / CONVERTED / "sound_0_4000.csv"
    pd.read_csv(path).assign(mood="happy").to_csv(path, index=False)
    _, warnings = am.validate_annotation(VTC_PRESENT)
    assert len(warnings) == 1 and "unknown column 'mood'" in warnings[0]


def test_validate_all(am):
    assert am.validate(threads=1) == ([], [])


def test_validate_subset(am):
    (am.project.path / ANNOTATIONS / "vtc_present" / CONVERTED / "sound_0_4000.csv").unlink()
    subset = am.annotations[am.annotations["recording_filename"] == "sound2.wav"]
    assert am.validate(subset, threads=1) == ([], [])
    assert len(am.validate(threads=1)[0]) == 1


def test_validate_skips_lines_without_annotation_file(am):
    am.annotations.loc[:, "annotation_filename"] = np.nan
    assert am.validate(threads=1) == ([], [])


def test_validate_requires_dataframe(am):
    with pytest.raises(AssertionError):
        am.validate(annotations=[VTC_PRESENT])


# ============================================================ write
def test_write_sorts_and_casts_index(am):
    am.annotations = pd.DataFrame([
        dict(VTC_PRESENT, imported_at="2024-02-01 00:00:00", time_seek=1.0),
        dict(VTC_PRESENT, recording_filename="sound2.wav", annotation_filename="sound2_0_4000.csv",
             imported_at="2024-01-01 00:00:00"),
    ])
    assert am.write() is am
    on_disk = pd.read_csv(am.project.path / METADATA_FOLDER / "annotations.csv")
    assert on_disk["annotation_filename"].tolist() == ["sound2_0_4000.csv", "sound_0_4000.csv"]
    assert on_disk["time_seek"].tolist() == [0, 1]
    assert on_disk["time_seek"].dtype == np.int64


def test_write_fills_missing_time_values_with_zero(am):
    am.annotations = pd.DataFrame([dict(VTC_PRESENT, time_seek=np.nan)])
    am.write()
    assert pd.read_csv(am.project.path / METADATA_FOLDER / "annotations.csv")["time_seek"].tolist() == [0]


# ============================================================ subsets, remove_set
@pytest.fixture
def nested_sets(am):
    for s in ["parent", "parent/child", "parent/child/grandchild", "parent/other"]:
        (am.project.path / ANNOTATIONS / s / CONVERTED).mkdir(parents=True, exist_ok=True)
    (am.project.path / ANNOTATIONS / "parent" / "not_a_dir.txt").touch()
    return am


def test_get_subsets(nested_sets):
    assert sorted(nested_sets.get_subsets("parent")) == ["parent/child", "parent/other"]


def test_get_subsets_recursive(nested_sets):
    assert sorted(nested_sets.get_subsets("parent", recursive=True)) == ["parent/child", "parent/child/grandchild", "parent/other"]


def test_get_subsets_ignores_raw_and_converted(am):
    assert am.get_subsets("vtc_present") == []


def test_remove_set_deletes_converted_files_and_index(am):
    assert am.remove_set("vtc_present") is am
    assert not (am.project.path / ANNOTATIONS / "vtc_present" / CONVERTED).exists()
    assert am.annotations.shape[0] == 0
    assert pd.read_csv(am.project.path / METADATA_FOLDER / "annotations.csv").shape[0] == 0


def test_remove_set_keeps_raw_files(am):
    am.remove_set("vtc_rttm")
    assert (am.project.path / ANNOTATIONS / "vtc_rttm" / RAW / "example.rttm").exists()


def test_remove_set_without_converted_folder(am, caplog):
    with caplog.at_level(logging.INFO, logger="ChildProject.annotations"):
        am.remove_set("vtc_rttm")
    assert "could not delete" in caplog.text


def test_remove_set_recursive(nested_sets):
    nested_sets.remove_set("parent", recursive=True)
    for s in ["parent", "parent/child", "parent/child/grandchild", "parent/other"]:
        assert not (nested_sets.project.path / ANNOTATIONS / s / CONVERTED).exists()


def test_remove_set_non_recursive_keeps_subsets(nested_sets):
    nested_sets.remove_set("parent")
    assert (nested_sets.project.path / ANNOTATIONS / "parent/child" / CONVERTED).exists()


# ============================================================ rename_set
def test_rename_set_moves_files(am):
    am.rename_set("vtc_present", "renamed")
    assert not (am.project.path / ANNOTATIONS / "vtc_present" / CONVERTED).exists()
    assert (am.project.path / ANNOTATIONS / "renamed" / CONVERTED / "sound_0_4000.csv").exists()
    assert (am.project.path / ANNOTATIONS / "renamed" / METANNOTS).exists()


def test_rename_set_updates_index(am):
    am.rename_set("vtc_present", "renamed")
    assert am.annotations["set"].unique().tolist() == ["renamed"]
    assert pd.read_csv(am.project.path / METADATA_FOLDER / "annotations.csv")["set"].unique().tolist() == ["renamed"]


def test_rename_set_strips_trailing_separators(am):
    am.rename_set("vtc_present/", "renamed/")
    assert am.annotations["set"].unique().tolist() == ["renamed"]


def test_rename_set_into_namespace_of_existing_set(am):
    am.rename_set("vtc_present", "alice")  # 'alice' folder exists but holds no raw/converted folder
    assert (am.project.path / ANNOTATIONS / "alice" / CONVERTED / "sound_0_4000.csv").exists()


def test_rename_set_missing_folder(am):
    with pytest.raises(Exception, match="does not exists"):
        am.rename_set("invented", "renamed")


@pytest.mark.parametrize("target,folder", [("vtc_rttm", "raw"), ("alice/output", "raw")])
def test_rename_set_target_folders_exist(am, target, folder):
    with pytest.raises(Exception, match=f"{folder} folder .* already exists"):
        am.rename_set("vtc_present", target)


def test_rename_set_target_converted_exists(am):
    (am.project.path / ANNOTATIONS / "target" / CONVERTED).mkdir(parents=True)
    with pytest.raises(Exception, match="converted folder .* already exists"):
        am.rename_set("vtc_present", "target")


def test_rename_set_target_already_indexed(am):
    am.annotations = pd.concat([am.annotations, pd.DataFrame([dict(VTC_PRESENT, set="invented")])])
    with pytest.raises(Exception, match="'invented' set already exists in the index"):
        am.rename_set("vtc_present", "invented")


def test_rename_set_not_indexed(am):
    with pytest.raises(Exception, match="set 'vtc_rttm' have no indexed annotation"):
        am.rename_set("vtc_rttm", "renamed")


def test_rename_set_not_indexed_ignore_errors(am):
    am.rename_set("vtc_rttm", "renamed", ignore_errors=True)
    assert (am.project.path / ANNOTATIONS / "renamed" / RAW / "example.rttm").exists()


def test_rename_set_recursive(nested_sets):
    nested_sets.rename_set("parent", "moved", recursive=True)
    assert (nested_sets.project.path / ANNOTATIONS / "moved/child/grandchild" / CONVERTED).exists()
    assert (nested_sets.project.path / ANNOTATIONS / "moved/other" / CONVERTED).exists()


def test_rename_set_updates_merged_from(am):
    am.annotations = pd.concat([am.annotations, pd.DataFrame([
        dict(VTC_PRESENT, set="merged", merged_from="vtc_present,other"),
        dict(VTC_PRESENT, set="unrelated", merged_from="other"),
    ])])
    am.rename_set("vtc_present", "renamed")
    merged_from = am.annotations.set_index("set")["merged_from"]
    assert sorted(merged_from["merged"].split(",")) == ["other", "renamed"]
    assert merged_from["unrelated"] == "other"


# ============================================================ segments
def test_get_segments(am):
    segments = am.get_segments(am.annotations)
    converted = am.project.path / ANNOTATIONS / "vtc_present" / CONVERTED
    expected = sum(pd.read_csv(converted / f).shape[0] for f in ["sound_0_4000.csv", "sound2_0_4000.csv"])
    assert segments.shape[0] == expected
    assert set(segments["recording_filename"]) == {"sound.wav", "sound2.wav"}
    assert {"set", "range_onset", "range_offset", "segment_onset", "speaker_type"} <= set(segments.columns)


def test_get_segments_clips_to_ranges(am):
    annotations = am.annotations.copy()
    annotations["range_offset"] = 500
    segments = am.get_segments(annotations)
    assert segments["segment_offset"].max() <= 500
    assert segments["segment_onset"].min() >= 0


def test_get_segments_without_matching_segments(am):
    annotations = am.annotations.copy()
    annotations["range_onset"], annotations["range_offset"] = 50_000, 60_000
    segments = am.get_segments(annotations)
    assert segments.shape[0] == 0
    assert {"segment_onset", "segment_offset", "set"} <= set(segments.columns)


def test_get_segments_ignores_lines_without_file(am):
    annotations = am.annotations.copy()
    annotations["annotation_filename"] = np.nan
    assert am.get_segments(annotations).shape[0] == 0


def test_get_segments_requires_columns(am):
    with pytest.raises(MissingColumnsException):
        am.get_segments(am.annotations.drop(columns=["raw_filename"]))


def test_get_collapsed_segments(am):
    segments = am.get_collapsed_segments(am.annotations.copy())
    sound2 = segments[segments["recording_filename"] == "sound2.wav"]
    sound = segments[segments["recording_filename"] == "sound.wav"]
    # 'sound.wav' sorts before 'sound2.wav': sound2's segments are shifted by sound's 4000 ms
    assert sound["position"].unique().tolist() == [0]
    assert sound2["position"].unique().tolist() == [4000]
    assert sound2["segment_onset"].min() >= 4000


def test_get_collapsed_segments_does_not_modify_input(am):
    annotations = am.annotations.copy()
    am.get_collapsed_segments(annotations)
    assert "duration" not in annotations.columns


def test_get_collapsed_segments_requires_columns(am):
    with pytest.raises(MissingColumnsException):
        am.get_collapsed_segments(am.annotations.drop(columns=["set"]))


# ============================================================ clip_segments
SEGMENTS = pd.DataFrame({"segment_onset": [0, 100, 500, 900], "segment_offset": [50, 600, 700, 1000]})


def test_clip_segments():
    clipped = AnnotationManager.clip_segments(SEGMENTS.copy(), 200, 800)
    assert clipped.values.tolist() == [[200, 600], [500, 700]]


def test_clip_segments_does_not_modify_input():
    segments = SEGMENTS.copy()
    AnnotationManager.clip_segments(segments, 200, 800)
    pd.testing.assert_frame_equal(segments, SEGMENTS)


def test_clip_segments_requires_columns():
    with pytest.raises(MissingColumnsException):
        AnnotationManager.clip_segments(SEGMENTS[["segment_onset"]], 0, 1)


# ============================================================ intersection
INTERSECT = pd.DataFrame([
    {"recording_filename": "r", "set": "a", "range_onset": 0, "range_offset": 1000},
    {"recording_filename": "r", "set": "a", "range_onset": 2000, "range_offset": 3000},
    {"recording_filename": "r", "set": "b", "range_onset": 500, "range_offset": 2500},
    {"recording_filename": "q", "set": "a", "range_onset": 0, "range_offset": 1000},
])


def test_intersection():
    result = AnnotationManager.intersection(INTERSECT)
    assert sorted(result[["range_onset", "range_offset", "set"]].values.tolist()) == [
        [500, 1000, "a"], [500, 1000, "b"], [2000, 2500, "a"], [2000, 2500, "b"],
    ]
    assert result["recording_filename"].unique().tolist() == ["r"]


def test_intersection_restricted_sets():
    result = AnnotationManager.intersection(INTERSECT, sets=["a"])
    assert result.shape[0] == 3


def test_intersection_empty():
    assert AnnotationManager.intersection(INTERSECT[INTERSECT["recording_filename"] == "q"].assign(set="a").iloc[:0]).shape[0] == 0


def test_intersection_requires_columns():
    with pytest.raises(MissingColumnsException):
        AnnotationManager.intersection(INTERSECT.drop(columns=["set"]))


def test_intersection_on_imported_sets(long_am):
    long_am.import_annotations(pd.read_csv(EXAMPLES / "valid_raw_data" / "annotations" / "intersect.csv"), threads=1)
    intersection = AnnotationManager.intersection(
        long_am.annotations[long_am.annotations["set"].isin(["textgrid", "vtc_rttm"])]).convert_dtypes()
    columns = [c for c in intersection.columns if c not in ("imported_at", "package_version", "merged_from")]
    for set_name, truth in [("textgrid", "intersect_a.csv"), ("vtc_rttm", "intersect_b.csv")]:
        pd.testing.assert_frame_equal(
            standardize(intersection[intersection["set"] == set_name], columns).convert_dtypes(),
            standardize(pd.read_csv(TRUTH / truth), columns).convert_dtypes(),
            check_dtype=False,
        )


# ============================================================ get_within_ranges
@pytest.fixture
def chunked(am):
    am.annotations = pd.DataFrame([
        {"recording_filename": "sound.wav", "set": "matching", "range_onset": onset, "range_offset": onset + 500}
        for onset in range(0, 4000, 500)
    ])
    return am


RANGE = pd.DataFrame([{"recording_filename": "sound.wav", "range_onset": 1000, "range_offset": 3000}])


def test_get_within_ranges(chunked):
    matches = chunked.get_within_ranges(RANGE, ["matching"])
    assert matches[["range_onset", "range_offset"]].values.tolist() == [[1000, 1500], [1500, 2000], [2000, 2500], [2500, 3000]]


def test_get_within_ranges_clips_partial_annotations(chunked):
    matches = chunked.get_within_ranges(RANGE.assign(range_onset=1250), ["matching"])
    assert matches[["range_onset", "range_offset"]].values.tolist()[0] == [1250, 1500]


def test_get_within_ranges_all_sets_by_default(chunked):
    assert chunked.get_within_ranges(RANGE).shape[0] == 4


def test_get_within_ranges_unknown_set(chunked):
    with pytest.raises(ValueError, match="the following sets are missing from the annotations index: nope"):
        chunked.get_within_ranges(RANGE, ["nope"])


def test_get_within_ranges_no_match(chunked):
    result = chunked.get_within_ranges(RANGE.assign(recording_filename="sound2.wav"))
    assert isinstance(result, pd.DataFrame) and result.shape[0] == 0


def test_get_within_ranges_raise_on_missing_data(chunked):
    with pytest.raises(Exception, match=r"annotations from set 'matching' do not cover the whole selected range "
                                        r"for recording 'sound.wav', 3.000s covered instead of 4.000s"):
        chunked.get_within_ranges(RANGE.assign(range_offset=5000), ["matching"], "raise")


def test_get_within_ranges_warn_on_missing_data(chunked, caplog):
    with caplog.at_level(logging.WARNING, logger="ChildProject.annotations"):
        result = chunked.get_within_ranges(RANGE.assign(range_offset=5000), ["matching"], "warn")
    assert "do not cover the whole selected range" in caplog.text
    assert result.shape[0] == 6


def test_get_within_ranges_fully_covered_is_silent(chunked, caplog):
    with caplog.at_level(logging.WARNING, logger="ChildProject.annotations"):
        chunked.get_within_ranges(RANGE, ["matching"], "raise")
    assert caplog.records == []


def test_get_within_ranges_requires_columns(chunked):
    with pytest.raises(MissingColumnsException):
        chunked.get_within_ranges(RANGE.drop(columns=["range_onset"]))


# ============================================================ get_within_time_range
@pytest.fixture
def time_range(am):
    am.project.recordings = pd.read_csv(TEST_DATA / "time_range_recordings.csv")
    return am, pd.read_csv(TEST_DATA / "time_range_annotations.csv"), pd.read_csv(TRUTH / "time_range.csv")


def test_get_within_time_range_interval(time_range):
    am, annotations, truth = time_range
    matches = am.get_within_time_range(annotations, TimeInterval(datetime.datetime(1900, 1, 1, 9), datetime.datetime(1900, 1, 1, 20)))
    pd.testing.assert_frame_equal(standardize(matches, truth.columns), standardize(truth, truth.columns))


def test_get_within_time_range_start_end(time_range):
    am, annotations, truth = time_range
    matches = am.get_within_time_range(annotations, start_time="09:00", end_time="20:00")
    pd.testing.assert_frame_equal(standardize(matches, truth.columns), standardize(truth, truth.columns))


def test_get_within_time_range_over_midnight(time_range):
    am, annotations, _ = time_range
    matches = am.get_within_time_range(annotations, start_time="23:00", end_time="01:00")
    ones = matches[(matches["recording_filename"] == "1.wav") & (matches["set"] == "auto")]
    # 1.wav starts at 08:00 and lasts 24h: 23:00-01:00 is [15h, 17h]
    assert ones[["range_onset", "range_offset"]].values.tolist() == [[15 * 3600000, 17 * 3600000]]


@pytest.mark.parametrize("start,end,message", [("9am", "20:00", "start_time"), ("09:00", "8pm", "end_time")])
def test_get_within_time_range_invalid_times(time_range, start, end, message):
    am, annotations, _ = time_range
    with pytest.raises(ValueError, match=f"invalid value for {message}"):
        am.get_within_time_range(annotations, start_time=start, end_time=end)


def test_get_within_time_range_requires_interval(time_range):
    am, annotations, _ = time_range
    with pytest.raises(AssertionError, match="you must pass an interval"):
        am.get_within_time_range(annotations)


def test_get_within_time_range_interval_type(time_range):
    am, annotations, _ = time_range
    with pytest.raises(ValueError, match="interval must be a TimeInterval object"):
        am.get_within_time_range(annotations, interval=("09:00", "10:00"))


def test_get_within_time_range_no_match(time_range):
    am, annotations, _ = time_range
    result = am.get_within_time_range(annotations[annotations["set"] == "manual"], start_time="20:00", end_time="21:00")
    assert result.shape[0] == 0
    assert "range_onset_ts" not in result.columns


def test_get_within_time_range_ignores_unknown_start_time(time_range):
    am, annotations, _ = time_range
    am.project.recordings.loc[0, "start_time"] = "NA"
    matches = am.get_within_time_range(annotations, start_time="09:00", end_time="20:00")
    assert "1.wav" not in matches["recording_filename"].tolist()


def test_get_within_time_range_requires_columns(time_range):
    am, annotations, _ = time_range
    with pytest.raises(MissingColumnsException):
        am.get_within_time_range(annotations.drop(columns=["range_onset"]), start_time="09:00", end_time="20:00")


# ============================================================ get_segments_timestamps
ONE_SEGMENT = pd.DataFrame([{"recording_filename": "sound.wav", "segment_onset": 3600 * 1000, "segment_offset": 3600 * 1000 + 1000}])


def test_get_segments_timestamps(am):
    result = am.get_segments_timestamps(ONE_SEGMENT.copy())
    assert result.loc[0, "onset_time"] == datetime.datetime(2020, 4, 20, 10, 0, 0)
    assert result.loc[0, "offset_time"] == datetime.datetime(2020, 4, 20, 10, 0, 1)
    assert "start_time" not in result.columns and "date_iso" not in result.columns


def test_get_segments_timestamps_ignore_date(am):
    result = am.get_segments_timestamps(ONE_SEGMENT.copy(), ignore_date=True)
    assert result.loc[0, "onset_time"] == datetime.datetime(1900, 1, 1, 10, 0, 0)


def test_get_segments_timestamps_custom_columns(am):
    segments = ONE_SEGMENT.rename(columns={"segment_onset": "a", "segment_offset": "b"})
    result = am.get_segments_timestamps(segments, onset="a", offset="b")
    assert result.loc[0, "offset_time"] == datetime.datetime(2020, 4, 20, 10, 0, 1)


def test_get_segments_timestamps_replaces_existing_columns(am):
    result = am.get_segments_timestamps(ONE_SEGMENT.assign(start_time="23:00"))
    assert result.loc[0, "onset_time"] == datetime.datetime(2020, 4, 20, 10, 0, 0)


def test_get_segments_timestamps_does_not_modify_input(am):
    segments = ONE_SEGMENT.assign(start_time="23:00")
    am.get_segments_timestamps(segments)
    assert "start_time" in segments.columns


def test_get_segments_timestamps_unknown_recording(am):
    result = am.get_segments_timestamps(ONE_SEGMENT.assign(recording_filename="nope.wav"))
    assert pd.isna(result.loc[0, "onset_time"])


def test_get_segments_timestamps_requires_columns(am):
    with pytest.raises(MissingColumnsException):
        am.get_segments_timestamps(ONE_SEGMENT.drop(columns=["segment_offset"]))


# ============================================================ set_from_path
@pytest.mark.parametrize("path,expected", [
    ("annotations/set", "set"),
    ("annotations/set/", "set"),
    ("annotations/set/subset", "set/subset"),
    ("annotations/set/subset/converted", "set/subset"),
    ("annotations/set/subset/raw", "set/subset"),
])
def test_set_from_path(am, path, expected):
    assert am.set_from_path(os.path.join(am.project.path, path)) == expected


def test_set_from_path_outside_annotations(am):
    assert am.set_from_path(am.project.path / "metadata") is None


# ============================================================ infer_set_content_based_on_column_names
@pytest.mark.parametrize("columns,expected", [
    (["speaker_type"], {"has_speaker_type": "Y"}),
    (["transcription", "words"], {"has_transcription": "Y", "has_words": "Y"}),
    (["is_CT", "conv_count"], {"has_interactions": "Y"}),
    (["is_CT"], {}),
    (["addressee", "vcm_type"], {"has_addressee": "Y", "has_vcm_type": "Y"}),
    ([], {}),
])
def test_infer_set_content(columns, expected):
    assert AnnotationManager.infer_set_content_based_on_column_names(columns) == expected


@pytest.mark.parametrize("columns,field", [
    (["lena_conv_turn_type"], "has_interactions"),     # second accepted combination
    (["median_pitch_semitone"], "has_acoustics"),      # any acoustic column
])
def test_infer_set_content_alternative_combinations(columns, field):
    assert AnnotationManager.infer_set_content_based_on_column_names(columns) == {field: "Y"}
