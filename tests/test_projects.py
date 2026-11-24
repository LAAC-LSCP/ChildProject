import filecmp
import logging
import os
from pathlib import Path

import pandas as pd
import pytest

from ChildProject.projects import (
    CHILDREN_CSV,
    CONVERTED_RECORDINGS,
    METADATA_FOLDER,
    RAW_RECORDINGS,
    RECORDINGS_CSV,
    ChildProject,
)
from ChildProject.tables import MissingColumnsException

from conftest import TRUTH, VALID_DATASET

MONTH = 365.25 / 12


def write_profile(project, profile, rows):
    """create a converted profile index (original_filename, converted_filename, success)"""
    index = pd.DataFrame(rows, columns=["original_filename", "converted_filename", "success"])
    project.write_profile(profile, index)
    return index


# ============================================================ constructor
def test_init_defaults():
    project = ChildProject("some/where")
    assert project.path == Path("some/where")
    assert project.enforce_dtypes is True
    assert project.ignore_discarded is True
    assert project.recordings is None and project.children is None
    assert project.loaded is False


# ============================================================ read
def test_read_returns_self(valid_path):
    project = ChildProject(valid_path)
    assert project.read() is project


def test_read_marks_project_as_loaded(valid_path):
    project = ChildProject(valid_path).read()
    assert project.loaded is True


def test_read_index_follows_csv_line_numbers(project):
    # line 1 is the header, discarded line 5 is excluded
    assert project.recordings.index.tolist() == [2, 3, 4]
    assert project.children.index.tolist() == [2]


def test_read_sets_experiment(project):
    assert project.experiment == "test"


def test_read_experiment_defaults_to_folder_name_without_children(valid_path):
    (valid_path / METADATA_FOLDER / CHILDREN_CSV).write_text("experiment,child_id,child_dob\n")
    project = ChildProject(valid_path).read()
    assert project.experiment == "valid_raw_data"


@pytest.mark.parametrize("enforce,kind", [(True, "O"), (False, "i")])
def test_read_enforce_dtypes(enforce, kind):
    project = ChildProject(VALID_DATASET, enforce_dtypes=enforce).read()
    assert project.recordings["child_id"].dtype.kind == kind
    assert project.children["child_id"].dtype.kind == kind


@pytest.mark.parametrize("ignore,recs,chis,drecs,dchis", [(True, 3, 1, 1, 1), (False, 4, 2, 0, 0)])
def test_read_ignore_discarded(ignore, recs, chis, drecs, dchis):
    project = ChildProject(VALID_DATASET, ignore_discarded=ignore).read()
    assert project.recordings.shape[0] == recs
    assert project.children.shape[0] == chis
    assert project.discarded_recordings.shape[0] == drecs
    assert project.discarded_children.shape[0] == dchis


def test_read_discard_values_that_are_not_1_are_kept(project):
    # children.csv has discard='random text' for child 1, it is coerced to 0
    assert project.children["discard"].tolist() == ["0"]


def test_read_accumulates_confidential_metadata(project):
    assert project.children.loc[2, "child_dob"] == "2020-01-01"
    assert project.recordings.set_index("recording_filename").loc["sound.wav", "notes"] == "VERY confidential notes"


def test_read_without_accumulate_keeps_main_csv_values(valid_path):
    project = ChildProject(valid_path).read(accumulate=False)
    assert project.children.loc[2, "child_dob"] == "2018-01-01"


def test_read_accumulate_keeps_original_value_when_override_is_empty(project):
    # 1_very_confidential.csv only has notes for sound.wav, other notes stay as in recordings.csv
    notes = project.recordings.set_index("recording_filename")["notes"]
    assert pd.isna(notes["sound2.wav"])


def test_read_verbose_logs_overwritten_columns(valid_path, caplog):
    with caplog.at_level(logging.WARNING, logger="ChildProject.projects"):
        ChildProject(valid_path).read(verbose=True)
    assert "column(s) child_dob overwritten by" in caplog.text
    assert "column(s) notes overwritten by" in caplog.text


def test_read_without_verbose_is_silent(valid_path, caplog):
    with caplog.at_level(logging.WARNING, logger="ChildProject.projects"):
        ChildProject(valid_path).read()
    assert "overwritten" not in caplog.text


def test_accumulate_metadata_ambiguous_filenames(valid_path):
    sub = valid_path / METADATA_FOLDER / "children" / "sub"
    sub.mkdir()
    (sub / "0_test.csv").write_text("child_id,child_dob\n1,2019-01-01\n")
    with pytest.raises(Exception, match="ambiguous filenames detected"):
        ChildProject(valid_path).read()


def test_accumulate_metadata_without_folder(valid_path):
    for f in (valid_path / METADATA_FOLDER / "children").iterdir():
        f.unlink()
    (valid_path / METADATA_FOLDER / "children").rmdir()
    project = ChildProject(valid_path).read()
    assert project.children.loc[2, "child_dob"] == "2018-01-01"


def test_accumulate_metadata_with_empty_folder(valid_path):
    (valid_path / METADATA_FOLDER / "children" / "0_test.csv").unlink()
    project = ChildProject(valid_path).read()
    assert project.children.loc[2, "child_dob"] == "2018-01-01"


def test_accumulate_metadata_skips_broken_links(valid_path):
    (valid_path / METADATA_FOLDER / "children" / "0_test.csv").unlink()
    os.symlink(valid_path / "does_not_exist.csv", valid_path / METADATA_FOLDER / "children" / "0_test.csv")
    project = ChildProject(valid_path).read()
    assert project.children.loc[2, "child_dob"] == "2018-01-01"


# ============================================================ profiles
def test_write_profile_creates_index(project):
    index = write_profile(project, "test", [["sound.wav", "sound.flac", True]])
    path = project.path / CONVERTED_RECORDINGS / "test" / RECORDINGS_CSV
    assert path.exists()
    pd.testing.assert_frame_equal(pd.read_csv(path), index)


def test_write_profile_unwritable_index(project, caplog):
    write_profile(project, "test", [["sound.wav", "sound.flac", True]])
    path = project.path / CONVERTED_RECORDINGS / "test" / RECORDINGS_CSV
    path.chmod(0o444)
    try:
        with caplog.at_level(logging.WARNING, logger="ChildProject.projects"):
            assert project.write_profile("test", pd.DataFrame()) is None
        assert "Could not write to" in caplog.text
    finally:
        path.chmod(0o644)


def test_read_profile(project):
    write_profile(project, "test", [["sound.wav", "sound.flac", True]])
    index = project.read_profile("test")
    assert index.index.tolist() == [2]
    assert index.loc[2, "converted_filename"] == "sound.flac"
    assert str(index["original_filename"].dtype) == "string"


def test_get_profiles(project):
    write_profile(project, "one", [["sound.wav", "sound.wav", True]])
    write_profile(project, "nested/two", [["sound.wav", "sound.wav", True]])
    assert sorted(project._get_profiles()) == ["nested/two", "one"]


def test_get_profiles_none(project):
    assert project._get_profiles() == []


# ============================================================ recording paths
def test_get_recording_path_raw(project):
    assert project.get_recording_path("sound.wav") == project.path / RAW_RECORDINGS / "sound.wav"


def test_get_recording_path_profile(project):
    write_profile(project, "conv", [["sound.wav", "sound.flac", True]])
    assert project.get_recording_path("sound.wav", "conv") == project.path / CONVERTED_RECORDINGS / "conv" / "sound.flac"


def test_get_recording_path_not_converted(project):
    write_profile(project, "conv", [["sound.wav", "sound.flac", True]])
    assert project.get_recording_path("sound2.wav", "conv") is None


def test_get_converted_recording_filename_ignores_failed_conversions(project):
    write_profile(project, "conv", [["sound.wav", None, False]])
    assert project.get_converted_recording_filename("conv", "sound.wav") is None


def test_get_converted_recording_filename_is_cached(project):
    write_profile(project, "conv", [["sound.wav", "sound.flac", True]])
    assert project.get_converted_recording_filename("conv", "sound.wav") == "sound.flac"
    (project.path / CONVERTED_RECORDINGS / "conv" / RECORDINGS_CSV).unlink()
    assert project.get_converted_recording_filename("conv", "sound.wav") == "sound.flac"
    assert project.converted_recordings_hashtable[("conv", "sound.wav")] == "sound.flac"


def test_get_converted_recording_filename_caches_misses(project):
    write_profile(project, "conv", [["sound.wav", "sound.flac", True]])
    assert project.get_converted_recording_filename("conv", "other.wav") is None
    assert project.converted_recordings_hashtable[("conv", "other.wav")] is None


def test_recording_from_path(project):
    assert project.recording_from_path(project.path / RAW_RECORDINGS / "sub" / "a.wav") == os.path.join("sub", "a.wav")


def test_recording_from_path_outside_raw_folder(project):
    assert project.recording_from_path(project.path / "elsewhere.wav") is None


def test_recording_from_path_with_profile_not_implemented(project):
    with pytest.raises(NotImplementedError):
        project.recording_from_path(project.path / RAW_RECORDINGS / "sound.wav", profile="conv")


# ============================================================ get_recordings_from_list
def test_get_recordings_from_list_none_returns_all(project):
    assert project.get_recordings_from_list(None)["recording_filename"].tolist() == ["sound.wav", "video.mp4", "sound2.wav"]


def test_get_recordings_from_list_excludes_NA_filenames(project):
    project.recordings.loc[3, "recording_filename"] = "NA"
    assert project.get_recordings_from_list(None)["recording_filename"].tolist() == ["sound.wav", "sound2.wav"]


def test_get_recordings_from_list_by_name(project):
    assert project.get_recordings_from_list(["sound2.wav"])["recording_filename"].tolist() == ["sound2.wav"]


def test_get_recordings_from_list_by_path(project):
    paths = [project.path / RAW_RECORDINGS / "sound.wav", project.path / RAW_RECORDINGS / "video.mp4"]
    assert project.get_recordings_from_list(paths)["recording_filename"].tolist() == ["sound.wav", "video.mp4"]


def test_get_recordings_from_list_unknown_recording(project):
    with pytest.raises(ValueError, match=r"recordings \['nope.wav'\] were not found"):
        project.get_recordings_from_list(["sound.wav", "nope.wav"])


def test_get_recordings_from_list_does_not_modify_project(project):
    before = project.recordings.copy()
    project.get_recordings_from_list(["sound.wav"])
    pd.testing.assert_frame_equal(project.recordings, before)


# ============================================================ durations
def test_compute_recordings_duration(project):
    durations = project.compute_recordings_duration()
    assert durations.columns.tolist() == ["recording_filename", "duration"]
    assert durations["duration"].tolist() == [4000, 10120, 4000]
    assert durations["duration"].dtype == "int64"


def test_compute_recordings_duration_missing_file(project):
    (project.path / RAW_RECORDINGS / "sound2.wav").unlink()
    assert project.compute_recordings_duration()["duration"].tolist() == [4000, 10120, 0]


def _profile_with(project, recordings):
    """profile 'conv' where each converted file is a hard link to sound.wav (4 s)"""
    profile_dir = project.path / CONVERTED_RECORDINGS / "conv"
    profile_dir.mkdir(parents=True)
    rows = []
    for i, rec in enumerate(recordings):
        os.link(project.path / RAW_RECORDINGS / "sound.wav", profile_dir / f"{i}.wav")
        rows.append([rec, f"{i}.wav", True])
    write_profile(project, "conv", rows)


def test_compute_recordings_duration_from_profile(project):
    _profile_with(project, ["sound.wav", "video.mp4", "sound2.wav"])
    assert project.compute_recordings_duration(profile="conv")["duration"].tolist() == [4000, 4000, 4000]


def test_compute_recordings_duration_recording_absent_from_profile(project):
    _profile_with(project, ["sound.wav"])
    assert project.compute_recordings_duration(profile="conv")["duration"].tolist() == [4000, 0, 0]


# ============================================================ ages
@pytest.mark.parametrize("fmt,expected", [
    ("months", [110 / MONTH, 110 / MONTH, 111 / MONTH]),
    ("days", [110, 110, 111]),
    ("weeks", [110 / 7, 110 / 7, 111 / 7]),
    ("years", [110 / 365.25, 110 / 365.25, 111 / 365.25]),
])
def test_compute_ages(project, fmt, expected):
    ages = project.compute_ages(age_format=fmt)
    assert ages.index.tolist() == project.recordings.index.tolist()
    assert ages.tolist() == pytest.approx(expected)


def test_compute_ages_default_is_months(project):
    assert project.compute_ages().tolist() == pytest.approx(project.compute_ages(age_format="months").tolist())


def test_compute_ages_matches_truth(project):
    truth = pd.read_csv(TRUTH / "ages.csv").set_index("line")
    pd.testing.assert_series_equal(project.compute_ages(), truth["age"], check_names=False, check_dtype=False)


def test_compute_ages_unknown_format(project):
    with pytest.raises(ValueError, match="unknown format for age"):
        project.compute_ages(age_format="decades")


def test_compute_ages_invalid_dates_give_missing_age(project):
    recordings = project.recordings.copy()
    recordings.loc[2, "date_iso"] = "NA"
    ages = project.compute_ages(recordings=recordings)
    assert pd.isna(ages[2])
    assert ages[4] == pytest.approx(111 / MONTH)


def test_compute_ages_on_birth_date_is_zero(project):
    recordings = project.recordings.copy()
    recordings["date_iso"] = "2020-01-01"
    assert project.compute_ages(recordings=recordings, age_format="days").tolist() == [0, 0, 0]


def test_compute_ages_with_custom_children(project):
    children = pd.DataFrame({"child_id": ["1"], "child_dob": ["2020-04-10"]})
    assert project.compute_ages(children=children, age_format="days").tolist() == [10, 10, 11]


def test_compute_ages_unknown_child(project):
    children = pd.DataFrame({"child_id": ["2"], "child_dob": ["2020-04-10"]})
    assert project.compute_ages(children=children).isna().all()


def test_compute_ages_requires_columns(project):
    with pytest.raises(MissingColumnsException):
        project.compute_ages(recordings=project.recordings.drop(columns=["date_iso"]))
    with pytest.raises(MissingColumnsException):
        project.compute_ages(children=project.children.drop(columns=["child_dob"]))


def test_compute_ages_requires_dataframes(project):
    with pytest.raises(AssertionError):
        project.compute_ages(recordings=[1, 2])


# ============================================================ dict_summary
@pytest.fixture
def loaded(project):
    """project flagged as loaded so that dict_summary works on the in-memory metadata
    (read() never sets the flag itself, see test_read_marks_project_as_loaded)"""
    project.loaded = True
    return project


def test_dict_summary(project):
    assert project.dict_summary() == {
        "recordings": {
            "count": 3, "duration": 18120, "first_date": "2020-04-20", "last_date": "2020-04-21",
            "discarded": 1, "devices": {"usb": {"count": 3, "duration": 18120}},
        },
        "children": {
            "count": 1, "min_age": 110 / MONTH, "max_age": 111 / MONTH, "M": None, "F": None,
            "languages": {}, "monolingual": None, "multilingual": None, "normative": None,
            "non-normative": None,
        },
    }


def test_dict_summary_reads_project_if_needed(valid_path):
    assert ChildProject(valid_path).dict_summary()["recordings"]["count"] == 3


def test_dict_summary_uses_in_memory_metadata(project):
    project.recordings = project.recordings.iloc[:1]
    assert project.dict_summary()["recordings"]["count"] == 1


def test_dict_summary_without_duration(loaded):
    project = loaded
    project.recordings = project.recordings.drop(columns=["duration"])
    summary = project.dict_summary()["recordings"]
    assert summary["duration"] is None
    assert summary["devices"]["usb"]["duration"] is None


def test_dict_summary_ignores_NA_dates(loaded):
    project = loaded
    project.recordings.loc[2, "date_iso"] = "NA"
    project.recordings.loc[3, "date_iso"] = "NA"
    summary = project.dict_summary()["recordings"]
    assert (summary["first_date"], summary["last_date"]) == ("2020-04-21", "2020-04-21")


@pytest.fixture
def described_children(loaded):
    project = loaded
    project.children = pd.DataFrame({
        "experiment": ["test"] * 3,
        "child_id": ["1", "2", "3"],
        "child_dob": ["2020-01-01"] * 3,
        "child_sex": ["m", "F", "f"],
        "languages": ["eng 0.5;fra 0.5", "eng", "spa"],
        "monoling": ["N", "Y", "Y"],
        "normative": ["Y", "N", "Y"],
    })
    return project


def test_dict_summary_children_sex(described_children):
    summary = described_children.dict_summary()["children"]
    assert (summary["M"], summary["F"]) == (1, 2)


def test_dict_summary_children_languages(described_children):
    assert described_children.dict_summary()["children"]["languages"] == {"eng": 2, "fra": 1, "spa": 1}


def test_dict_summary_children_without_language_are_not_a_language(described_children):
    described_children.children.loc[2, "languages"] = None
    assert described_children.dict_summary()["children"]["languages"] == {"eng": 2, "fra": 1}


def test_dict_summary_children_monolingual_normative(described_children):
    summary = described_children.dict_summary()["children"]
    assert (summary["monolingual"], summary["multilingual"]) == (2, 1)
    assert (summary["normative"], summary["non-normative"]) == (2, 1)


def test_dict_summary_single_language_column(loaded):
    project = loaded
    project.children = project.children.assign(language="spa (main)")
    assert project.dict_summary()["children"]["languages"] == {"spa": 1}


# ============================================================ write_recordings
NEW_RECORDINGS = dict(
    experiment=["test", "test"], child_id=["1", "1"], recording_filename=["sound.wav", "sound2.wav"],
    start_time=["NA", "05:32"], date_iso=["2021-03-05", "2023-11-29"], recording_type=["audio", "audio"],
    recording_device_type=["usb", "unknown"],
)


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def test_write_recordings_uninitialized_returns_none(valid_path):
    assert ChildProject(valid_path).write_recordings() is None


def test_write_recordings_refuses_invalid_dataset(project):
    project.recordings = pd.DataFrame(dict(NEW_RECORDINGS, experiment=["other", "other"])).convert_dtypes()
    with pytest.raises(ValueError, match="Dataset is not validating"):
        project.write_recordings()


def test_write_recordings_skip_validation(project):
    project.recordings = pd.DataFrame(dict(NEW_RECORDINGS, experiment=["other", "other"])).convert_dtypes()
    project.write_recordings(skip_validation=True)
    assert read_csv(project.path / METADATA_FOLDER / RECORDINGS_CSV)["experiment"].tolist()[:2] == ["other", "other"]


def test_write_recordings_keeps_discarded(project):
    project.recordings = pd.DataFrame(NEW_RECORDINGS).convert_dtypes()
    project.write_recordings()
    written = read_csv(project.path / METADATA_FOLDER / RECORDINGS_CSV)
    assert written["recording_filename"].tolist() == ["sound.wav", "sound2.wav", "sound3.wav"]
    assert written["discard"].tolist() == ["0", "0", "1"]


def test_write_recordings_drops_discarded(project):
    project.recordings = pd.DataFrame(NEW_RECORDINGS).convert_dtypes()
    project.write_recordings(keep_discarded=False)
    written = read_csv(project.path / METADATA_FOLDER / RECORDINGS_CSV)
    assert written["recording_filename"].tolist() == ["sound.wav", "sound2.wav"]
    assert written["discard"].tolist() == ["0", "0"]


def test_write_recordings_round_trip(project):
    project.recordings = pd.DataFrame(NEW_RECORDINGS).convert_dtypes()
    project.write_recordings()
    reread = ChildProject(project.path).read(accumulate=False)
    assert reread.recordings["date_iso"].tolist() == ["2021-03-05", "2023-11-29"]
    assert reread.discarded_recordings["recording_filename"].tolist() == ["sound3.wav"]


def test_write_recordings_returns_written_dataframe(project):
    written = project.write_recordings()
    assert written.shape[0] == 4


# ============================================================ write_children
def test_write_children_uninitialized_returns_none(valid_path):
    assert ChildProject(valid_path).write_children() is None


@pytest.mark.parametrize("children", [
    {"experiment": ["other"], "child_id": ["1"], "child_dob": ["2015-05-04"]},  # experiment mismatch
    {"experiment": ["test"], "child_id": ["1"], "child_dob": ["2025-05-04"]},   # negative age
])
def test_write_children_refuses_invalid_dataset(project, children):
    project.children = pd.DataFrame(children).convert_dtypes()
    with pytest.raises(ValueError, match="Dataset is not validating"):
        project.write_children()


def test_write_children_skip_validation(project):
    project.children = pd.DataFrame({"experiment": ["test"], "child_id": ["1"], "child_dob": ["2025-05-04"]}).convert_dtypes()
    project.write_children(skip_validation=True)
    assert read_csv(project.path / METADATA_FOLDER / CHILDREN_CSV)["child_dob"].tolist()[0] == "2025-05-04"


def test_write_children_keeps_discarded(project):
    project.children = pd.DataFrame({"experiment": ["test"], "child_id": ["1"], "child_dob": ["2017-05-04"]}).convert_dtypes()
    project.write_children()
    written = read_csv(project.path / METADATA_FOLDER / CHILDREN_CSV)
    assert written["child_id"].tolist() == ["1", "2"]
    assert written["discard"].tolist() == ["0", "1"]


def test_write_children_drops_discarded(project):
    project.children = pd.DataFrame({"experiment": ["test"], "child_id": ["1"], "child_dob": ["2017-05-04"]}).convert_dtypes()
    project.write_children(keep_discarded=False)
    assert read_csv(project.path / METADATA_FOLDER / CHILDREN_CSV)["child_id"].tolist() == ["1"]


# ============================================================ add_project_file
SOURCE = "metadata/children/0_test.csv"


@pytest.mark.parametrize("dst_file,file_type,dst_path", [
    ("rec008.wav", "recording", "recordings/raw/rec008.wav"),
    ("rec008", "recording", "recordings/raw/rec008"),
    (Path("rec008.wav"), "recording", "recordings/raw/rec008.wav"),
    ("loc1/RA5/rec008.wav", "recording", "recordings/raw/loc1/RA5/rec008.wav"),
    ("metrics.csv", "extra", "extra/metrics.csv"),
    ("metadata.xlsx", "metadata", "metadata/metadata.xlsx"),
    ("README.md", "raw", "README.md"),
    ("scripts/new_subfolder/any_script.py", "raw", "scripts/new_subfolder/any_script.py"),
])
def test_add_project_file(project, dst_file, file_type, dst_path):
    src = project.path / SOURCE
    project.add_project_file(src, dst_file, file_type)
    assert filecmp.cmp(src, project.path / dst_path, shallow=False)


def test_add_project_file_accepts_str_source(project):
    project.add_project_file(str(project.path / SOURCE), "new.csv", "extra")
    assert (project.path / "extra" / "new.csv").exists()


def test_add_project_file_returns_project(project):
    assert project.add_project_file(project.path / SOURCE, "new.csv", "extra") is project


def test_add_project_file_existing_destination(project):
    with pytest.raises(FileExistsError):
        project.add_project_file(project.path / SOURCE, "sound.wav", "recording")


def test_add_project_file_overwrite(project):
    project.add_project_file(project.path / SOURCE, "children.csv", "metadata", overwrite=True)
    assert filecmp.cmp(project.path / SOURCE, project.path / METADATA_FOLDER / CHILDREN_CSV, shallow=False)


def test_add_project_file_missing_source(project):
    with pytest.raises(FileNotFoundError):
        project.add_project_file(project.path / "made_up_file.txt", "sound5.wav", "recording")


@pytest.mark.parametrize("dst_file,file_type", [("/etc/sound.wav", "recording"), ("../other_place", "raw")])
def test_add_project_file_outside_dataset(project, dst_file, file_type):
    with pytest.raises(AssertionError):
        project.add_project_file(project.path / SOURCE, dst_file, file_type)


def test_add_project_file_unknown_type(project):
    with pytest.raises(ValueError, match="unknown file_type"):
        project.add_project_file(project.path / SOURCE, "sound5.wav", "made_up_type")


def test_add_project_file_refuses_to_overwrite_symlink(project):
    os.symlink(project.path / SOURCE, project.path / "linked.md")
    with pytest.raises(AssertionError, match="annexed data"):
        project.add_project_file(project.path / SOURCE, "linked.md", "raw", overwrite=True)


def test_add_project_file_warns_on_extension_change(project, caplog):
    with caplog.at_level(logging.WARNING, logger="ChildProject.projects"):
        project.add_project_file(project.path / SOURCE, "rec.wav", "recording")
    assert "have different file extensions" in caplog.text


# ============================================================ remove_project_file
@pytest.mark.parametrize("file,file_type,path", [
    ("sound.wav", "recording", "recordings/raw/sound.wav"),
    ("children.csv", "metadata", "metadata/children.csv"),
    ("fake_readme.txt", "extra", "extra/fake_readme.txt"),
    ("README.txt", "raw", "README.txt"),
])
def test_remove_project_file(project, file, file_type, path):
    (project.path / path).parent.mkdir(exist_ok=True)
    (project.path / path).touch()
    project.remove_project_file(file, file_type)
    assert not (project.path / path).exists()


def test_remove_project_file_missing(project):
    with pytest.raises(FileNotFoundError):
        project.remove_project_file("sound6.wav", "recording")


@pytest.mark.parametrize("file", ["/sound.wav", "../../../../sound.wav"])
def test_remove_project_file_outside_dataset(project, file):
    with pytest.raises(AssertionError):
        project.remove_project_file(file, "recording")


def test_remove_project_file_unknown_type(project):
    with pytest.raises(ValueError, match="unknown file_type"):
        project.remove_project_file("sound.wav", "made_up_type")


def test_remove_project_file_refuses_symlink(project):
    os.symlink(project.path / SOURCE, project.path / "linked.md")
    with pytest.raises(AssertionError, match="annexed data"):
        project.remove_project_file("linked.md", "raw")


# ============================================================ rename_recording
def test_rename_recording_moves_file(project):
    project.rename_recording("sound.wav", "new_sound.wav")
    assert not (project.path / RAW_RECORDINGS / "sound.wav").exists()
    assert (project.path / RAW_RECORDINGS / "new_sound.wav").exists()


def test_rename_recording_returns_new_name(project):
    assert project.rename_recording("sound.wav", "new_sound.wav") == "new_sound.wav"


def test_rename_recording_updates_index_in_memory_and_on_disk(project):
    before = project.recordings.set_index("recording_filename").loc["sound.wav"]
    project.rename_recording("sound.wav", "new_sound.wav")
    after = project.recordings.set_index("recording_filename").loc["new_sound.wav"]
    pd.testing.assert_series_equal(before, after, check_names=False)
    on_disk = read_csv(project.path / METADATA_FOLDER / RECORDINGS_CSV)
    assert "new_sound.wav" in on_disk["recording_filename"].tolist()
    assert "sound.wav" not in on_disk["recording_filename"].tolist()


def test_rename_recording_to_existing_name(project):
    with pytest.raises(AssertionError, match="already exists"):
        project.rename_recording("sound.wav", "sound2.wav")


def test_rename_recording_missing_file_warns_but_updates_index(project, caplog):
    (project.path / RAW_RECORDINGS / "sound.wav").unlink()
    with caplog.at_level(logging.WARNING, logger="ChildProject.projects"):
        project.rename_recording("sound.wav", "new_sound.wav")
    assert "could not be renamed" in caplog.text
    assert "new_sound.wav" in project.recordings["recording_filename"].tolist()


def test_rename_recording_not_indexed_still_moves_file(project):
    (project.path / RAW_RECORDINGS / "orphan.wav").touch()
    project.rename_recording("orphan.wav", "adopted.wav")
    assert (project.path / RAW_RECORDINGS / "adopted.wav").exists()


def test_rename_recording_updates_profiles(project):
    write_profile(project, "conv", [["sound.wav", "sound.flac", True], ["sound2.wav", "sound2.flac", True]])
    write_profile(project, "other", [["sound2.wav", "sound2.flac", True]])
    project.rename_recording("sound.wav", "new_sound.wav")
    assert project.read_profile("conv")["original_filename"].tolist() == ["new_sound.wav", "sound2.wav"]
    assert project.read_profile("other")["original_filename"].tolist() == ["sound2.wav"]


# ============================================================ documentation
def test_read_documentation(project):
    doc = project.read_documentation()
    truth = pd.read_csv(TRUTH / "docs.csv", dtype_backend="numpy_nullable")
    pd.testing.assert_frame_equal(
        doc[truth.columns].reset_index(drop=True), truth, check_dtype=False
    )


def test_read_documentation_combines_tables(project):
    (project.path / "docs" / "recordings.csv").write_text("variable,description\nnotes,rec notes\n")
    doc = project.read_documentation()
    assert sorted(doc["table"].unique()) == ["children", "recordings"]


def test_read_documentation_without_docs(project):
    for f in (project.path / "docs").iterdir():
        f.unlink()
    doc = project.read_documentation()
    assert isinstance(doc, pd.DataFrame)
    assert doc.shape[0] == 0
