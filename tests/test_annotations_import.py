"""AnnotationManager.import_annotations / _import_annotation"""
import logging
import re
from functools import partial
from pathlib import Path

import pandas as pd
import pytest

from ChildProject import __version__
from ChildProject.annotations import ANNOTATIONS, CONVERTED, AnnotationManager
from ChildProject.converters import VtcConverter
from ChildProject.projects import EXTRA, METADATA_FOLDER
from ChildProject.tables import MissingColumnsException

from ChildProject.projects import ChildProject

from conftest import EXAMPLES, TEST_DATA, TRUTH, VALID_DATASET, copy_dataset, import_static, static_segments

INPUT = EXAMPLES / "valid_raw_data" / "annotations" / "input.csv"
EXPECTED_WARNINGS_AFTER_IMPORT = [
    "Metadata files for sets ['vtc_rttm'] could not be found, they should be created as annotations/<set>/metannots.yml",
    "Metadata file content for sets ['old_its'] could not be found, it may be downloaded from a remote with the "
    f"command `datalad get {Path('annotations/**/metannots.yml')}`",
    "Metadata files for sets contain the following unknown fields ['invented', 'random_field'] which can be found in "
    "the metadata for sets ['alice/output', 'textgrid']",
]


def standardize(df, columns):
    df = df[list(columns)]
    return df.sort_index(axis=1).sort_values(list(columns)).reset_index(drop=True)


def adapt_path(message):
    """express paths found in truth messages with the separator of the current OS"""
    return re.sub(r"(?:[a-zA-Z]:)?(?:[\\/][^:\s]+)+", lambda x: str(Path(x.group())), message)


def one_line(**overrides):
    line = {"set": "vtc_rttm", "recording_filename": "sound.wav", "time_seek": 0, "raw_filename": "example.rttm",
            "range_onset": 0, "range_offset": 4000, "format": "vtc_rttm"}
    line.update(overrides)
    return pd.DataFrame([line])


def converted_path(am, row):
    return am.project.path / ANNOTATIONS / row["set"] / CONVERTED / row["annotation_filename"]


def long_manager(path):
    project = ChildProject(path)
    am = AnnotationManager(project)
    project.recordings["duration"] = 100_000_000
    return am


@pytest.fixture(scope="module")
def imported_template(tmp_path_factory):
    """the full example input is imported once per module (slow), each test then gets its own copy"""
    path = copy_dataset(VALID_DATASET, tmp_path_factory.mktemp("imported") / "valid_raw_data")
    long_manager(path).import_annotations(pd.read_csv(INPUT), threads=1)
    return path


@pytest.fixture
def imported(imported_template, tmp_path):
    """fresh copy of a valid dataset where all the annotations of examples/.../input.csv were imported"""
    return long_manager(copy_dataset(imported_template, tmp_path / "valid_raw_data"))


# ============================================================ full importation of the example input
def test_import_adds_one_index_line_per_input_line(long_am):
    before = long_am.annotations.shape[0]
    long_am.import_annotations(pd.read_csv(INPUT), threads=1)
    assert long_am.annotations.shape[0] == before + pd.read_csv(INPUT).shape[0]


def test_import_returns_imported_lines_and_no_errors(long_am):
    imported, errors = long_am.import_annotations(pd.read_csv(INPUT), threads=1)
    assert imported.shape[0] == pd.read_csv(INPUT).shape[0]
    assert errors is None or errors.shape[0] == 0


def test_import_creates_converted_files(imported):
    for row in imported.annotations.to_dict(orient="records"):
        assert converted_path(imported, row).exists(), row["annotation_filename"]


def test_import_writes_the_index(imported):
    on_disk = pd.read_csv(imported.project.path / METADATA_FOLDER / "annotations.csv")
    assert on_disk.shape[0] == imported.annotations.shape[0]


def test_imported_annotations_validate(imported):
    assert imported.validate(threads=1) == ([], [])


def test_imported_index_reads_without_error(imported):
    errors, warnings = imported.read()
    assert errors == []
    assert warnings == EXPECTED_WARNINGS_AFTER_IMPORT


def test_import_fills_generated_columns(imported):
    row = imported.annotations[imported.annotations["set"] == "textgrid"].iloc[0]
    assert row["annotation_filename"] == "sound_0_10000.csv"
    assert row["package_version"] == __version__
    assert re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", row["imported_at"])


@pytest.mark.parametrize("set_name", ["eaf_basic", "textgrid", "eaf_solis"])
def test_imported_segments_match_truth(imported, set_name):
    annotations = imported.annotations[imported.annotations["set"] == set_name]
    segments = imported.get_segments(annotations)
    truth = pd.read_csv(TRUTH / f"{set_name}.csv", dtype_backend="numpy_nullable")
    columns = set(truth.columns)
    pd.testing.assert_frame_equal(standardize(segments, columns), standardize(truth, columns),
                                  check_exact=False, rtol=1e-3)


def test_import_with_pool(long_am):
    input_annotations = pd.read_csv(INPUT)
    input_annotations = input_annotations[input_annotations["format"] != "cha"]
    imported, _ = long_am.import_annotations(input_annotations, threads=2)
    assert imported.shape[0] == input_annotations.shape[0]


# ============================================================ input checks
@pytest.mark.parametrize("line,column,value,message", [
    (0, "range_onset", -10, "range_onset must be greater or equal to 0"),
    (3, "range_offset", 1970000, "range_offset must be greater than range_onset"),
    (3, "range_offset", 1980000, "range_offset must be greater than range_onset"),
])
def test_import_rejects_invalid_ranges(am, line, column, value, message):
    input_annotations = pd.read_csv(INPUT)
    input_annotations.iloc[line, input_annotations.columns.get_loc(column)] = value
    with pytest.raises(AssertionError, match=message):
        am.import_annotations(input_annotations)


def test_import_rejects_range_beyond_duration(am):
    with pytest.raises(AssertionError, match="range_offset must be smaller than the duration of the recording"):
        am.import_annotations(one_line(range_offset=4001))


def test_import_without_duration_skips_duration_check(am):
    am.project.recordings = am.project.recordings.drop(columns=["duration"])
    imported, _ = am.import_annotations(one_line(range_offset=10_000_000, range_onset=1980000), threads=1)
    assert imported.shape[0] == 1


def test_import_rejects_unknown_recordings(am):
    with pytest.raises(ValueError, match="not referenced in the metadata:\nnope.wav"):
        am.import_annotations(one_line(recording_filename="nope.wav"))


def test_import_requires_columns(am):
    with pytest.raises(MissingColumnsException):
        am.import_annotations(one_line().drop(columns=["raw_filename"]))


def test_import_rejects_overlapping_input(long_am):
    with pytest.raises(ValueError, match="the ranges given to import have overlaps on indexes"):
        long_am.import_annotations(pd.read_csv(TEST_DATA / "input_invalid.csv"))


def test_import_overlapping_input_imports_nothing(long_am):
    before = long_am.annotations.copy()
    with pytest.raises(ValueError):
        long_am.import_annotations(pd.read_csv(TEST_DATA / "input_invalid.csv"))
    pd.testing.assert_frame_equal(long_am.annotations, before)


def test_import_rejects_duration_overflow(imported):
    with pytest.raises(AssertionError):
        imported.import_annotations(pd.read_csv(TEST_DATA / "input_import_duration_overflow.csv"))


def test_import_does_not_modify_input(long_am):
    input_annotations = pd.read_csv(INPUT)
    before = input_annotations.copy()
    long_am.import_annotations(input_annotations, threads=1)
    pd.testing.assert_frame_equal(input_annotations, before)


# ============================================================ re-importation and overlaps with the index
def compare_to_truth(result, truth_file, error_paths=False):
    truth = pd.read_csv(TRUTH / truth_file)
    if error_paths:
        truth["error"] = truth["error"].apply(adapt_path)
    pd.testing.assert_frame_equal(truth.reset_index(drop=True),
                                  result.drop(["imported_at", "package_version"], axis=1).reset_index(drop=True),
                                  check_like=True, check_dtype=False)


def test_reimport_without_overwrite_imports_only_new_ranges(imported):
    result, _ = imported.import_annotations(pd.read_csv(TEST_DATA / "input_reimport.csv"), threads=1)
    compare_to_truth(result, "imp_reimport_no_ow.csv")


def test_reimport_without_overwrite_reports_existing_files(imported):
    _, errors = imported.import_annotations(pd.read_csv(TEST_DATA / "input_reimport.csv"), threads=1)
    compare_to_truth(errors, "err_reimport_no_ow.csv", error_paths=True)


def test_reimport_with_overwrite(imported):
    result, errors = imported.import_annotations(pd.read_csv(TEST_DATA / "input_reimport.csv"),
                                                 overwrite_existing=True, threads=1)
    compare_to_truth(result, "imp_reimport_ow.csv")
    assert errors is None or errors.shape[0] == 0


def test_reimport_with_overwrite_does_not_duplicate_index_lines(imported):
    before = imported.annotations.shape[0]
    imported.import_annotations(pd.read_csv(TEST_DATA / "input_reimport.csv"), overwrite_existing=True, threads=1)
    assert imported.annotations.shape[0] == before + 1  # only the 'should always work' line is new


def test_reimport_with_overwrite_logs_warning(imported, caplog):
    with caplog.at_level(logging.WARNING, logger="ChildProject.annotations"):
        imported.import_annotations(one_line(set="textgrid", raw_filename="example.TextGrid", format="TextGrid",
                                             range_offset=10000), overwrite_existing=True, threads=1)
    assert "will be overwritten" in caplog.text


def test_import_overlapping_existing_annotations(imported):
    result, errors = imported.import_annotations(pd.read_csv(TEST_DATA / "input_importoverlaps.csv"), threads=1)
    compare_to_truth(result, "imp_overlap.csv")
    compare_to_truth(errors, "err_overlap.csv", error_paths=True)


def test_import_errors_are_exported_to_extra(imported):
    imported.import_annotations(pd.read_csv(TEST_DATA / "input_importoverlaps.csv"), threads=1)
    exports = list((imported.project.path / EXTRA).glob("errors_import_*.csv"))
    assert len(exports) == 1
    assert pd.read_csv(exports[0]).shape[0] == 2


def test_index_stays_valid_after_partial_reimport(imported):
    imported.import_annotations(pd.read_csv(TEST_DATA / "input_importoverlaps.csv"), threads=1)
    assert imported.read()[0] == []
    assert imported.validate(threads=1) == ([], [])


# ============================================================ conversion
def test_import_unknown_format(am):
    _, errors = am.import_annotations(one_line(format="unknown_format"), threads=1)
    assert "file format 'unknown_format' unknown" in errors.iloc[0]["error"]


def test_import_missing_raw_file(am):
    _, errors = am.import_annotations(one_line(raw_filename="missing.rttm"), threads=1)
    assert errors.shape[0] == 1
    assert "FileNotFoundError" in errors.iloc[0]["error"]


@pytest.mark.parametrize("source,count", [("namibie_aiku_20160714_1", 5), ("other_recording", 0)])
def test_import_applies_filter(long_am, source, count):
    imported, _ = long_am.import_annotations(one_line(range_onset=0, range_offset=100_000_000, filter=source), threads=1)
    assert pd.read_csv(converted_path(long_am, imported.iloc[0])).shape[0] == count


def test_import_time_seek_shifts_segments(am):
    segments = pd.DataFrame({"segment_onset": [0, 500], "segment_offset": [100, 600], "speaker_type": ["CHI", "FEM"]})
    imported, _ = am.import_annotations(one_line(time_seek=1000, range_onset=1000, range_offset=4000),
                                        import_function=partial(static_segments, segments), threads=1)
    shifted = pd.read_csv(converted_path(am, imported.iloc[0]))
    assert shifted[["segment_onset", "segment_offset"]].values.tolist() == [[1000, 1100], [1500, 1600]]


def test_import_clips_segments_to_range(long_am):
    imported, _ = long_am.import_annotations(
        one_line(range_onset=1982000, range_offset=1984000, filter="namibie_aiku_20160714_1"), threads=1)
    segments = pd.read_csv(converted_path(long_am, imported.iloc[0]))
    assert segments["segment_onset"].min() >= 1982000
    assert segments["segment_offset"].max() <= 1984000


def test_import_sorts_segments(long_am, conversation_segments):
    shuffled = conversation_segments.sample(frac=1, random_state=0)
    imported, _ = import_static(long_am, shuffled, "shuffled")
    segments = pd.read_csv(converted_path(long_am, imported.iloc[0]))
    assert segments["segment_onset"].is_monotonic_increasing


def test_import_with_custom_function(am):
    def custom_function(filename):
        df = pd.read_csv(filename, sep=" ", names=["type", "file", "chnl", "tbeg", "tdur", "ortho", "stype", "name", "conf", "unk"])
        df["segment_onset"] = 1000 * df["tbeg"].astype(int)
        df["segment_offset"] = (1000 * (df["tbeg"] + df["tdur"])).astype(int)
        df["speaker_type"] = df["name"].map(VtcConverter.SPEAKER_TYPE_TRANSLATION)
        return df[["segment_onset", "segment_offset", "speaker_type"]]

    imported, _ = am.import_annotations(one_line(format="custom"), import_function=custom_function, threads=1)
    assert imported.shape[0] == 1
    assert am.validate(threads=1)[0] == []


def test_import_function_not_returning_dataframe(am):
    imported, errors = am.import_annotations(one_line(), import_function=lambda f: [1, 2], threads=1)
    assert imported.shape[0] == 0
    assert errors is None or errors.shape[0] == 0
    assert "vtc_rttm" not in am.annotations["set"].tolist()


def test_import_function_returning_no_column(am):
    imported, _ = am.import_annotations(one_line(), import_function=lambda f: pd.DataFrame(), threads=1)
    segments = pd.read_csv(converted_path(am, imported.iloc[0]))
    assert segments.shape[0] == 0
    assert "segment_onset" in segments.columns


def test_import_rejects_incorrect_segment_types(am):
    incorrect_types = pd.DataFrame({
        "segment_onset": ["0", "10", "20"],
        "segment_offset": ["5", "15", "25"],
        "speaker_type": ["CHI", "FEM", "MAN"],
    })
    _, errors = import_static(am, incorrect_types, "incorrect_types", range_offset=4000)
    assert errors is not None and errors.shape[0] == 1


def test_import_eaf_with_new_tiers(long_am):
    long_am.add_annotation_file(TEST_DATA / "eaf_any_tier.eaf", "any.eaf", "any_tier", False)
    imported, _ = long_am.import_annotations(
        one_line(set="any_tier", raw_filename="any.eaf", format="eaf", range_offset=1_000_000),
        new_tiers=["newtier", "newtier2"], threads=1)
    assert "newtier" in pd.read_csv(converted_path(long_am, imported.iloc[0])).columns


def test_import_non_thread_safe_converter_runs_on_one_thread(long_am, caplog):
    long_am.add_annotation_file(TEST_DATA / "vandam.cha", "vandam.cha", "cha", False)
    with caplog.at_level(logging.WARNING, logger="ChildProject.annotations"):
        imported, _ = long_am.import_annotations(
            one_line(set="cha", raw_filename="vandam.cha", format="cha", range_offset=100_000_000), threads=4)
    assert "do not support multithread importation" in caplog.text
    assert imported.shape[0] == 1


def test_import_missing_format_is_recorded_as_NA(am, conversation_segments):
    imported, _ = am.import_annotations(one_line(format=None), import_function=partial(static_segments, conversation_segments), threads=1)
    assert imported.iloc[0]["format"] == "NA"


def test_import_logs_outdated_merged_sets(long_am, caplog):
    long_am.annotations["merged_from"] = pd.NA
    long_am.annotations = pd.concat([long_am.annotations, pd.DataFrame([{
        "set": "merged", "recording_filename": "sound.wav", "time_seek": 0, "range_onset": 0, "range_offset": 100,
        "merged_from": "vtc_rttm", "imported_at": "2000-01-01 00:00:00", "annotation_filename": "x.csv"}])])
    with caplog.at_level(logging.WARNING, logger="ChildProject.annotations"):
        long_am.import_annotations(one_line(range_onset=1980000, range_offset=1990000), threads=1)
    assert "set merged is outdated" in caplog.text
