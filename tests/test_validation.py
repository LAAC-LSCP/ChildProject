"""validation of datasets: ChildProject.validate and the pydantic validators of the metadata"""
import os

import pandas as pd
import pytest
from pydantic import __version__ as pydantic_version

from ChildProject.annotations import AnnotationManager
from ChildProject.projects import METADATA_FOLDER, RECORDINGS_CSV, CHILDREN_CSV, ChildProject

PYDANTIC_URL = "https://errors.pydantic.dev/" + ".".join(pydantic_version.split(".")[:-1])


def append_line(path, line):
    with open(path, "a") as f:
        f.write(line + "\n")


def edit_csv(path, **changes):
    """set columns of the first data line of a metadata csv"""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col, value in changes.items():
        df.loc[0, col] = value
    df.to_csv(path, index=False)


# ============================================================ valid dataset
def test_valid_dataset_has_no_error(valid_path):
    errors, _ = ChildProject(valid_path).validate()
    assert errors == []


def test_valid_dataset_warns_about_unindexed_files(valid_path):
    _, warnings = ChildProject(valid_path).validate()
    assert warnings == ["files {'unindexed.mp3'} not indexed in recording_filename column"]


def test_validate_reads_the_dataset(valid_path):
    project = ChildProject(valid_path)
    project.validate()
    assert project.recordings is not None


def test_validate_stores_errors_and_warnings_on_project(invalid_path):
    project = ChildProject(invalid_path)
    errors, warnings = project.validate()
    assert project.errors == errors and project.warnings == warnings


# ============================================================ invalid dataset: one test per expected issue
@pytest.fixture
def invalid_errors(invalid_path):
    project = ChildProject(invalid_path)
    errors, warnings = project.validate()
    am = AnnotationManager(project)
    return errors + am.errors, warnings + am.warnings, invalid_path


def expected_invalid_errors(path):
    return [
        str(path / "metadata" / "children.csv") + ": Duplicated values when it should be unique for column child_id, values {'1'} on lines {2, 3} appear multiple times",
        str(path / "metadata" / "recordings.csv") + ": \n3 validation errors for RecordingModel\n"
        "recording_type\n  Field required [type=missing, input_value={'experiment': 'test', 'c... 2, 'duration': 1990000}, input_type=dict]\n"
        f"    For further information visit {PYDANTIC_URL}/v/missing\n"
        "recording_device_type\n  String should match pattern 'lena|usb|olympus|babylogger|izyrec|unknown' [type=string_pattern_mismatch, input_value='USB', input_type=str]\n"
        f"    For further information visit {PYDANTIC_URL}/v/string_pattern_mismatch\n"
        "noisy_setting\n  Input should be a valid boolean, unable to interpret input [type=bool_parsing, input_value=2, input_type=int]\n"
        f"    For further information visit {PYDANTIC_URL}/v/bool_parsing",
        "'recording_filename' values ['test_1_20200918.mp3', 'test_1_is_not_here.wav'] in recordings table on lines [2, 5] cannot be found in the filesystem.",
        "Age at recording is negative in recordings on line 3 (-15.4 months). Check date_iso for that recording and child_dob for the corresponding child.",
        "Age at recording is negative in recordings on line 4 (-15.4 months). Check date_iso for that recording and child_dob for the corresponding child.",
        "Age at recording is negative in recordings on line 5 (-15.4 months). Check date_iso for that recording and child_dob for the corresponding child.",
        "duplicate reference to annotations/vtc_rttm/converted/sound_1980000_1990000.csv (appears 2 times)",
        "annotation index does not verify range_offset > range_onset >= 0 for set <ranges>, annotation filename <sound_0_300000.csv>",
        "annotation index has an offset higher than recorded duration of the audio <textgrid>, annotation filename <sound_0_40000000.csv>",
    ]


@pytest.mark.parametrize("i", range(9))
def test_invalid_dataset_reports_error(invalid_errors, i):
    errors, _, path = invalid_errors
    assert expected_invalid_errors(path)[i] in errors


def test_invalid_dataset_reports_no_other_error(invalid_errors):
    errors, _, path = invalid_errors
    assert sorted(errors) == sorted(expected_invalid_errors(path))


def test_invalid_dataset_warnings(invalid_errors):
    _, warnings, _ = invalid_errors
    assert sorted(warnings) == sorted([
        "Metadata files for sets ['alice', 'ranges', 'textgrid', 'vtc_rttm'] could not be found, they should be created as annotations/<set>/metannots.yml",
        "files {'test_1_2020091.mp3'} not indexed in recording_filename column",
    ])


# ============================================================ individual checks
def test_validate_missing_recording_file(valid_path):
    (valid_path / "recordings" / "raw" / "sound2.wav").unlink()
    errors, _ = ChildProject(valid_path).validate()
    assert errors == ["'recording_filename' values ['sound2.wav'] in recordings table on lines [4] cannot be found in the filesystem."]


def test_validate_ignore_recordings(valid_path):
    (valid_path / "recordings" / "raw" / "sound2.wav").unlink()
    errors, warnings = ChildProject(valid_path).validate(ignore_recordings=True)
    assert errors == [] and warnings == []


def test_validate_profile_checks_converted_recordings(valid_path):
    """validating a profile that does not exist must not pass silently"""
    errors, _ = ChildProject(valid_path).validate(profile="does_not_exist")
    assert any("does_not_exist" in e for e in errors)


def test_validate_experiment_must_be_unique(valid_path):
    edit_csv(valid_path / METADATA_FOLDER / RECORDINGS_CSV, experiment="other")
    errors, _ = ChildProject(valid_path).validate(ignore_recordings=True)
    assert len(errors) == 1
    assert errors[0].startswith("Column <experiment> must be unique across the dataset")
    assert "2 different values were found" in errors[0]


def test_validate_unknown_child_id(valid_path):
    edit_csv(valid_path / METADATA_FOLDER / RECORDINGS_CSV, child_id="42")
    errors, _ = ChildProject(valid_path).validate(ignore_recordings=True)
    assert len(errors) == 1
    assert errors[0].startswith("child_ids ")
    assert "'42'" in errors[0]


def test_validate_unknown_child_id_message_is_readable(valid_path):
    """the message must list values and lines, not pandas object representations"""
    edit_csv(valid_path / METADATA_FOLDER / RECORDINGS_CSV, child_id="42")
    errors, _ = ChildProject(valid_path).validate(ignore_recordings=True)
    assert "child_ids ['42'] in recordings table on lines [2] cannot be found in the children table." in errors[0]
    assert "Index(" not in errors[0]


def test_validate_negative_age(valid_path):
    # date_iso is overridden by confidential metadata, so validate the main csv only
    edit_csv(valid_path / METADATA_FOLDER / RECORDINGS_CSV, date_iso="2017-12-01")
    errors, _ = ChildProject(valid_path).validate(ignore_recordings=True, accumulate=False)
    assert errors == ["Age at recording is negative in recordings on line 2 (-1.0 months). Check date_iso for that recording and child_dob for the corresponding child."]


def test_validate_without_accumulate_ignores_confidential_metadata(valid_path):
    # the confidential date of birth (2020-01-01) is later than sound2's recording date once modified
    edit_csv(valid_path / METADATA_FOLDER / "children" / "0_test.csv", child_dob="2020-04-21")
    project = ChildProject(valid_path)
    assert project.validate(ignore_recordings=True, accumulate=True)[0] != []
    assert project.validate(ignore_recordings=True, accumulate=False)[0] == []


def test_validate_current_metadata_uses_in_memory_tables(project):
    (project.path / METADATA_FOLDER / RECORDINGS_CSV).unlink()  # would break a re-read
    project.recordings.loc[2, "experiment"] = "other"
    errors, _ = project.validate(ignore_recordings=True, current_metadata=True)
    assert any(e.startswith("Column <experiment> must be unique") for e in errors)


def test_validate_current_metadata_detects_invalid_values(project):
    project.recordings.loc[2, "recording_type"] = "film"
    errors, _ = project.validate(ignore_recordings=True, current_metadata=True)
    assert len(errors) == 1
    assert "recording_type\n  String should match pattern 'audio|video'" in errors[0]


def test_validate_unknown_column_warning(valid_path):
    path = valid_path / METADATA_FOLDER / CHILDREN_CSV
    df = pd.read_csv(path, dtype=str, keep_default_na=False).assign(favourite_toy="bear")
    df.to_csv(path, index=False)
    _, warnings = ChildProject(valid_path).validate(ignore_recordings=True)
    assert len(warnings) == 1
    assert "unknown column 'favourite_toy' in children" in warnings[0]


# ============================================================ metadata validators (one rule per test)
@pytest.mark.parametrize("column,value,message", [
    ("recording_type", "", "recording_type"),
    ("recording_type", "VIDEO", "recording_type\n  String should match pattern 'audio|video'"),
    ("recording_device_type", "phone", "recording_device_type\n  String should match pattern"),
    ("date_iso", "20-04-2020", "date_iso"),
    ("start_time", "25:00", "start_time"),
    ("duration", "-5", "duration"),
    ("noisy_setting", "2", "noisy_setting"),
])
def test_recording_validator_rejects(valid_path, column, value, message):
    edit_csv(valid_path / METADATA_FOLDER / RECORDINGS_CSV, **{column: value})
    errors, _ = ChildProject(valid_path).validate(ignore_recordings=True, accumulate=False)
    assert any(message in e for e in errors), errors


@pytest.mark.parametrize("column,value", [
    ("recording_type", "video"),
    ("recording_type", "audio"),
    ("start_time", "NA"),
    ("start_time", "9:05:03"),
    ("recording_device_type", "unknown"),
])
def test_recording_validator_accepts(valid_path, column, value):
    edit_csv(valid_path / METADATA_FOLDER / RECORDINGS_CSV, **{column: value})
    errors, _ = ChildProject(valid_path).validate(ignore_recordings=True, accumulate=False)
    assert errors == []


@pytest.mark.parametrize("column,value", [
    ("child_dob", "01/01/2018"),
    ("child_sex", "X"),
    ("monoling", "maybe"),
    ("normative", "maybe"),
    ("dob_accuracy", "century"),
])
def test_child_validator_rejects(valid_path, column, value):
    edit_csv(valid_path / METADATA_FOLDER / CHILDREN_CSV, **{column: value})
    errors, _ = ChildProject(valid_path).validate(ignore_recordings=True, accumulate=False)
    assert any(column in e for e in errors), errors


def test_child_validator_requires_columns(valid_path):
    path = valid_path / METADATA_FOLDER / CHILDREN_CSV
    pd.read_csv(path, dtype=str).drop(columns=["child_dob"]).to_csv(path, index=False)
    (valid_path / METADATA_FOLDER / "children" / "0_test.csv").unlink()
    errors, _ = ChildProject(valid_path).validate(ignore_recordings=True)
    assert any("child_dob\n  Field required" in e for e in errors), errors
