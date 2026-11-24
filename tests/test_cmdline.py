"""command line interface (ChildProject.cmdline), run in-process through main()"""
import json
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

import ChildProject.cmdline as cli
from ChildProject import __version__
from ChildProject.annotations import AnnotationManager
from ChildProject.projects import ChildProject

from conftest import EXAMPLES, INVALID_DATASET, TEST_DATA, import_static

_register = cli.register_pipeline


def _register_once(name, cls):
    """main() registers the pipelines on every call; argparse refuses duplicated sub-commands"""
    if name not in cli.subparsers.choices:
        _register(name, cls)


@pytest.fixture
def run(monkeypatch, caplog):
    """run `child-project <argv...>` in-process; returns the logged messages"""
    monkeypatch.setattr(cli, "register_pipeline", _register_once)

    def _run(*argv):
        caplog.clear()
        monkeypatch.setattr(sys, "argv", ["child-project", *map(str, argv)])
        with caplog.at_level(logging.INFO, logger="ChildProject"):
            cli.main()
        return caplog.messages

    return _run


def output(messages):
    return "\n".join(messages)


# ============================================================ helpers of the module
def test_set_encoder_serializes_sets():
    assert json.loads(json.dumps({"a": {1, 2}}, cls=cli.SetEncoder)) == {"a": [1, 2]}


def test_set_encoder_rejects_other_objects():
    with pytest.raises(TypeError):
        json.dumps({"a": object()}, cls=cli.SetEncoder)


@pytest.mark.parametrize("doc,summary", [("first line\nsecond", "first line"), (None, ""), ("", "")])
def test_get_doc_summary(doc, summary):
    assert cli.get_doc_summary(doc) == summary


@pytest.mark.parametrize("level,info,no_info", [(logging.INFO, True, False), (logging.WARNING, False, True)])
def test_log_filters_split_info_messages(level, info, no_info):
    record = logging.LogRecord("x", level, "f", 1, "msg", None, None)
    assert cli.InfoFilter().filter(record) is info
    assert cli.NoInfoFilter().filter(record) is no_info


def test_perform_validation_raises_on_errors():
    with pytest.raises(ValueError, match="dataset validation failed"):
        cli.perform_validation(ChildProject(INVALID_DATASET))


def test_perform_validation_can_proceed_despite_errors(caplog):
    with caplog.at_level(logging.WARNING, logger="ChildProject"):
        cli.perform_validation(ChildProject(INVALID_DATASET), require_success=False)
    assert "Proceeding despite errors" in caplog.text


# ============================================================ main
def test_version(run, capsys):
    with pytest.raises(SystemExit) as exc:
        run("--version")
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_verbose_flag(run, capsys):
    run("--verbose", "interpreter")
    assert capsys.readouterr().out.strip() == sys.executable


def test_interpreter(run, capsys):
    run("interpreter")
    assert capsys.readouterr().out.strip() == sys.executable


def test_unknown_command(run):
    with pytest.raises(SystemExit) as exc:
        run("does-not-exist")
    assert exc.value.code == 2


# ============================================================ init
def test_init_creates_structure(run, tmp_path):
    run("init", tmp_path / "new")
    for folder in ["recordings/raw", "metadata", "extra", "scripts", "annotations"]:
        assert (tmp_path / "new" / folder).is_dir()
    assert (tmp_path / "new" / "README.md").exists()


def test_init_writes_required_columns(run, tmp_path):
    run("init", tmp_path / "new")
    recordings = pd.read_csv(tmp_path / "new" / "metadata" / "recordings.csv")
    assert list(recordings.columns) == [c.name for c in ChildProject.RECORDINGS_COLUMNS if c.required]
    children = pd.read_csv(tmp_path / "new" / "metadata" / "children.csv")
    assert list(children.columns) == [c.name for c in ChildProject.CHILDREN_COLUMNS if c.required]


def test_init_refuses_non_empty_folder(run, tmp_path):
    (tmp_path / "file.txt").touch()
    with pytest.raises(ValueError, match="not empty"):
        run("init", tmp_path)


def test_init_force(run, tmp_path):
    (tmp_path / "file.txt").touch()
    run("init", tmp_path, "--force")
    assert (tmp_path / "metadata" / "children.csv").exists()


# ============================================================ validate
def test_validate_valid_dataset(run, valid_path):
    assert "validation successfully completed with 1 warning(s)." in run("validate", valid_path)


def test_validate_invalid_dataset(run, invalid_path, caplog):
    with pytest.raises(SystemExit) as exc:
        run("validate", invalid_path)
    assert exc.value.code == 1
    assert any(m.startswith("validation failed, 9 error(s)") for m in caplog.messages)


def test_validate_ignore_recordings(run, valid_path):
    (valid_path / "recordings" / "raw" / "sound2.wav").unlink()
    assert "validation successfully completed with 0 warning(s)." in run("validate", valid_path, "--ignore-recordings")


@pytest.mark.parametrize("by_path", [False, True])
def test_validate_annotation_sets(run, valid_path, by_path):
    target = valid_path / "annotations" / "vtc_present" if by_path else "vtc_present"
    assert "validation successfully completed with 1 warning(s)." in run("validate", valid_path, "--annotations", target, "--threads", 1)


def test_validate_annotation_set_errors(run, valid_path, caplog):
    (valid_path / "annotations" / "vtc_present" / "converted" / "sound_0_4000.csv").unlink()
    with pytest.raises(SystemExit):
        run("validate", valid_path, "--annotations", "vtc_present", "--threads", 1)
    assert any("error while trying to read sound_0_4000.csv" in m for m in caplog.messages)


def test_validate_unknown_annotation_set(run, valid_path, caplog):
    with pytest.raises(SystemExit):
        run("validate", valid_path, "--annotations", "nope", "--threads", 1)
    assert "the following annotation sets are not indexed: nope" in caplog.messages


# ============================================================ overview / sets-metadata / explain
def test_overview(run, valid_path):
    text = output(run("overview", valid_path))
    assert "3 recordings with 0.01 hours 3 locally (1 discarded)" in text
    assert "date range :\x1b[0m 2020-04-20 to 2020-04-21" in text
    assert "usb (0.01h 3/3 locally);" in text
    assert "age range :\x1b[0m 3.6mo to 3.6mo" in text
    assert "8.0s automated  VTC       1 2024-04-07" in text


def test_overview_json(run, valid_path):
    data = json.loads(run("overview", valid_path, "--format", "json")[-1])
    assert data["test"]["recordings"]["count"] == 3
    assert "vtc_present" in data["test"]["annotations"]


def test_overview_children_details(run, valid_path):
    path = valid_path / "metadata" / "children.csv"
    pd.read_csv(path, dtype=str).assign(child_sex="F", language="eng", monoling="Y", normative="N").to_csv(path, index=False)
    text = output(run("overview", valid_path))
    assert "sex distribution :\x1b[0m 0M 1F" in text
    assert "languages :\x1b[0m eng 1;" in text
    assert "monolinguality :\x1b[0m 1mono 0multi" in text
    assert "normativity :\x1b[0m 0norm 1non-norm" in text


def test_overview_ages_in_years(run, valid_path):
    path = valid_path / "metadata" / "children" / "0_test.csv"
    path.write_text("child_id,child_dob\n1,2015-01-01\n")
    assert "5.3yo to 5.3yo" in output(run("overview", valid_path))


def test_overview_several_datasets(run, valid_path, invalid_path, caplog):
    messages = run("overview", valid_path, invalid_path)
    assert any(f"### test ({valid_path}) ###" in m for m in messages)
    assert any(m.startswith(f"{invalid_path}: [ValueError]") for m in caplog.messages)


def test_sets_metadata(run, valid_path):
    assert run("sets-metadata", valid_path)[-1] == (
        "duration    method algo version       date transcr\n"
        "    8000 automated  VTC       1 2024-04-07         \x1b[94mvtc_present\x1b[0m\n")


def test_sets_metadata_human_readable(run, valid_path):
    assert "    8.0s automated" in run("sets-metadata", valid_path, "--human-readable")[-1]


def test_sets_metadata_csv(run, valid_path):
    assert run("sets-metadata", valid_path, "--format", "csv")[-1].splitlines()[1].startswith("vtc_present,")


def test_sets_metadata_several_datasets(run, valid_path, invalid_path, caplog):
    messages = run("sets-metadata", valid_path, invalid_path)
    assert any("### test" in m for m in messages)
    assert any(m.startswith(f"{invalid_path}: [ValueError]") for m in caplog.messages)


@pytest.mark.parametrize("variable,table,expected", [
    ("child_dob", "children", "The date of birth of the child!"),            # from docs/children.csv
    ("date_iso", "recordings", "date in which recording was started in ISO"),  # from the recordings columns
    ("speaker_type", "annotations", "class of speaker"),                      # from the segments columns
])
def test_explain(run, valid_path, variable, table, expected):
    text = output(run("explain", valid_path, variable))
    assert f"Matching documentation for '{variable}'" in text
    assert f"table\x1b[0m: {table}" in text
    assert expected in text


def test_explain_prints_plain_description(run, valid_path):
    text = output(run("explain", valid_path, "child_dob"))
    assert "The date of birth of the child!" in text


def test_explain_unknown_variable(run, valid_path):
    assert run("explain", valid_path, "non-existent-variable") == ["Could not find any documentation for variable 'non-existent-variable'"]


def test_explain_without_documentation_files(run, valid_path):
    shutil.rmtree(valid_path / "docs")
    assert "Matching documentation for 'date_iso'" in output(run("explain", valid_path, "date_iso"))


# ============================================================ recordings
def test_compute_durations(run, valid_path):
    recordings = valid_path / "metadata" / "recordings.csv"
    pd.read_csv(recordings).drop(columns=["duration"]).to_csv(recordings, index=False)
    run("compute-durations", valid_path)
    assert pd.read_csv(recordings)["duration"].tolist()[:3] == [4000, 10120, 4000]


def test_compute_durations_existing_column(run, valid_path):
    assert "The 'duration' column already exists, aborting the procces" in run("compute-durations", valid_path)


def test_compute_durations_force(run, valid_path):
    recordings = valid_path / "metadata" / "recordings.csv"
    df = pd.read_csv(recordings)
    df["duration"] = 1
    df.to_csv(recordings, index=False)
    run("compute-durations", valid_path, "--force")
    assert pd.read_csv(recordings)["duration"].tolist()[:3] == [4000, 10120, 4000]


def test_compute_durations_does_not_write_confidential_metadata(run, valid_path):
    run("compute-durations", valid_path, "--force")
    assert "VERY confidential notes" not in pd.read_csv(valid_path / "metadata" / "recordings.csv")["notes"].tolist()[1:]
    assert ChildProject(valid_path).read(accumulate=False).children.loc[2, "child_dob"] == "2018-01-01"


def test_rename_recording(run, valid_path):
    run("rename-recording", valid_path, "--filename", "sound.wav", "--new-filename", "renamed.wav")
    assert (valid_path / "recordings" / "raw" / "renamed.wav").exists()
    assert "renamed.wav" in AnnotationManager(ChildProject(valid_path)).annotations["recording_filename"].tolist()


def test_compare_recordings(run, valid_path):
    text = output(run("compare-recordings", valid_path, "sound.wav", "sound2.wav", "--interval", 10))
    assert "RESULTS :\ndivergence score = " in text
    assert "duration of the audio is too short" in text
    assert "number of values (1639) in the sample is low" in text


def test_compare_recordings_different_durations(run, valid_path):
    recordings = valid_path / "metadata" / "recordings.csv"
    df = pd.read_csv(recordings)
    df.loc[df["recording_filename"] == "sound2.wav", "duration"] = 3000
    df.to_csv(recordings, index=False)
    assert "the 2 audio files have different durations" in output(run("compare-recordings", valid_path, "sound.wav", "sound2.wav"))


def test_compare_recordings_unknown_recording(run, valid_path):
    with pytest.raises(ValueError, match="nope.wav was not found in the indexed recordings"):
        run("compare-recordings", valid_path, "sound.wav", "nope.wav")


def test_compare_recordings_without_duration(run, valid_path):
    recordings = valid_path / "metadata" / "recordings.csv"
    pd.read_csv(recordings).drop(columns=["duration"]).to_csv(recordings, index=False)
    assert "RESULTS :" in output(run("compare-recordings", valid_path, "sound.wav", "sound2.wav", "--interval", 10))


# ============================================================ annotations
def index(path):
    return pd.read_csv(path / "metadata" / "annotations.csv")


def test_import_annotations_from_csv(run, valid_path):
    run("import-annotations", valid_path, "--annotations", EXAMPLES / "valid_raw_data" / "annotations" / "input_short.csv", "--threads", 1)
    assert set(index(valid_path)["set"]) >= {"textgrid", "eaf_basic", "vtc_rttm", "alice/output", "new_its"}


def test_import_annotations_from_arguments(run, valid_path):
    run("import-annotations", valid_path, "--set", "textgrid", "--recording_filename", "sound.wav", "--time_seek", 0,
        "--raw_filename", "example.TextGrid", "--range_onset", 0, "--range_offset", 1000, "--format", "TextGrid")
    assert index(valid_path).set_index("set").loc["textgrid", "annotation_filename"] == "sound_0_1000.csv"


def test_import_annotations_reports_failures(run, valid_path, caplog):
    args = ["import-annotations", valid_path, "--annotations", EXAMPLES / "valid_raw_data" / "annotations" / "input_short.csv",
            "--threads", 1]
    run(*args)
    run(*args)
    assert "The importation failed for 10 entry/ies" in caplog.messages


def test_import_annotations_overwrite(run, valid_path, caplog):
    args = ["import-annotations", valid_path, "--annotations", EXAMPLES / "valid_raw_data" / "annotations" / "input_short.csv",
            "--threads", 1]
    run(*args)
    run(*args, "--overwrite-existing")
    assert not any("importation failed" in m for m in caplog.messages)


def test_automated_import(run, valid_path):
    shutil.copy(valid_path / "annotations" / "vtc_rttm" / "raw" / "example.rttm", valid_path / "annotations" / "vtc_rttm" / "raw" / "sound.rttm")
    run("automated-import", valid_path, "--set", "vtc_rttm", "--format", "vtc_rttm", "--threads", 1)
    imported = index(valid_path)
    assert imported[imported["set"] == "vtc_rttm"]["raw_filename"].tolist() == ["sound.rttm"]


def test_automated_import_logs_missing_raw_files(run, valid_path, caplog):
    run("automated-import", valid_path, "--set", "vtc_rttm", "--format", "vtc_rttm", "--threads", 1)
    assert "The importation failed for 3 entry/ies" in caplog.messages


def test_automated_import_requires_duration(run, valid_path):
    recordings = valid_path / "metadata" / "recordings.csv"
    pd.read_csv(recordings).drop(columns=["duration"]).to_csv(recordings, index=False)
    with pytest.raises(ValueError, match="Column <duration> is needed"):
        run("automated-import", valid_path, "--set", "vtc_rttm", "--format", "vtc_rttm")


def test_derive_annotations(run, valid_path):
    run("derive-annotations", valid_path, "conversations", "-i", "vtc_present", "-o", "vtc/conversations", "--threads", 1)
    assert (index(valid_path)["set"] == "vtc/conversations").sum() == 2


def test_derive_annotations_reports_failures(run, valid_path, caplog):
    run("derive-annotations", valid_path, "acoustics", "-i", "vtc_present", "-o", "acoustics", "--threads", 1)
    assert "The derivation failed for 2 entry/ies" in caplog.messages


@pytest.fixture
def mergeable(long_am):
    input_annotations = pd.read_csv(EXAMPLES / "valid_raw_data" / "annotations" / "input.csv")
    long_am.import_annotations(input_annotations[input_annotations["set"].isin(["vtc_rttm", "alice/output"])], threads=1)
    return long_am.project.path


def test_merge_annotations(run, mergeable):
    run("merge-annotations", mergeable, "--left-set", "vtc_rttm", "--right-set", "alice/output", "--left-columns",
        "speaker_type", "--right-columns", "words,phonemes", "--output-set", "alice", "--threads", 1)
    assert (index(mergeable)["set"] == "alice").sum() == 2


def test_intersect_annotations(run, valid_path, tmp_path):
    run("intersect-annotations", valid_path, "--destination", tmp_path / "inter.csv", "--sets", "vtc_present", "vtc_present")
    assert pd.read_csv(tmp_path / "inter.csv").shape[0] == 2


def test_intersect_annotations_custom_index(run, valid_path, tmp_path):
    run("intersect-annotations", valid_path, "--destination", tmp_path / "inter.csv", "--sets", "textgrid", "vtc_rttm",
        "--annotations", EXAMPLES / "valid_raw_data" / "annotations" / "intersect.csv")
    # textgrid [3800, 3920] and [7400, 7520] intersected with vtc_rttm [0, 7500]: 2 ranges x 2 sets
    assert pd.read_csv(tmp_path / "inter.csv").shape[0] == 4


def test_remove_annotations(run, valid_path):
    run("remove-annotations", valid_path, "--set", "vtc_present")
    assert index(valid_path).shape[0] == 0
    assert not (valid_path / "annotations" / "vtc_present" / "converted").exists()


def test_rename_annotations(run, valid_path):
    run("rename-annotations", valid_path, "--set", "vtc_present", "--new-set", "vtc_renamed")
    assert set(index(valid_path)["set"]) == {"vtc_renamed"}


# ============================================================ pipelines
def test_process(run, valid_path):
    run("process", valid_path, "basic", "test", "--format", "wav", "--sampling", 16000, "--codec", "pcm_s16le")
    assert pd.read_csv(valid_path / "recordings" / "converted" / "test" / "recordings.csv")["success"].all()


def test_process_standard(run, valid_path):
    run("process", valid_path, "standard", "--recordings", "sound.wav")
    assert (valid_path / "recordings" / "converted" / "standard" / "sound.wav").exists()


def test_sampler(run, valid_path, tmp_path):
    run("sampler", valid_path, tmp_path / "samples", "periodic", "--length", 100, "--period", 400, "--offset", 200)
    assert len(list((tmp_path / "samples").glob("segments_*.csv"))) == 1


def test_zooniverse_help(run, capsys):
    with pytest.raises(SystemExit) as exc:
        run("zooniverse", "-h")
    assert exc.value.code == 0
    assert "extract-chunks" in capsys.readouterr().out


def test_zooniverse_reset_orphan_subjects(run, tmp_path):
    shutil.copy(TEST_DATA / "chunks_test.csv", tmp_path / "chunks.csv")
    run("zooniverse", "reset-orphan-subjects", "--chunks", tmp_path / "chunks.csv")


def test_eaf_builder(run, tmp_path):
    run("eaf-builder", "--destination", tmp_path / "eaf", "--segments", "tests/truth/segments_metrics.csv",
        "--eaf-type", "random", "--template", "basic")
    assert (tmp_path / "eaf" / "sound_random_basic.eaf").exists()


def test_anonymize(run, valid_path):
    run("anonymize", valid_path, "--input-set", "new_its", "--output-set", "new_its_anonymized")
    assert (valid_path / "annotations" / "new_its_anonymized" / "raw" / "example_lena_new.its").exists()


def test_metrics(run, valid_path, tmp_path):
    run("metrics", valid_path, tmp_path / "metrics.csv", "aclew", "--vtc", "vtc_present")
    assert pd.read_csv(tmp_path / "metrics.csv")["recording_filename"].tolist() == ["sound.wav", "video.mp4", "sound2.wav"]


def test_metrics_specification(run, long_am, lena_segments, tmp_path):
    import_static(long_am, lena_segments, "specs_its", range_offset=100_000_000, fmt="its")
    spec = yaml.safe_load((TEST_DATA / "parameters_metrics.yml").read_text())
    spec.update(path=str(long_am.project.path), destination=str(tmp_path / "specs.csv"))
    (tmp_path / "spec.yml").write_text(yaml.dump(spec))
    run("metrics-specification", tmp_path / "spec.yml")
    assert (tmp_path / "specs.csv").exists()


@pytest.fixture
def conversations(long_am, conversation_segments):
    import_static(long_am, conversation_segments, "custom_conv")
    return long_am.project.path


def test_conversations_summary(run, conversations, tmp_path):
    run("conversations-summary", conversations, tmp_path / "conv.csv", "--set", "custom_conv", "standard")
    assert pd.read_csv(tmp_path / "conv.csv").shape[0] == 3


def test_conversations_specification(run, conversations, tmp_path):
    spec = yaml.safe_load((TEST_DATA / "conversations_parameters.yml").read_text())
    spec["parameters"].update(path=str(conversations), destination=str(tmp_path / "conv.csv"))
    (tmp_path / "spec.yml").write_text(yaml.dump(spec))
    run("conversations-specification", tmp_path / "spec.yml")
    assert pd.read_csv(tmp_path / "conv.csv").shape[0] == 3


# ============================================================ installed entry point
def test_console_script_is_installed():
    import subprocess
    result = subprocess.run(["child-project", "--version"], capture_output=True, text=True)
    assert result.returncode == 0
    assert __version__ in result.stdout
