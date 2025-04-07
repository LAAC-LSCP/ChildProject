import subprocess
import pytest
import os
import shutil
from pathlib import Path

PATH = os.path.join("output", "cli")

def cli(cmd):
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    exit_code = process.wait()
    # replace Windows line feeds with posix ones for testing purposes
    stdout = stdout.replace(b'\r\n', b'\n').decode()
    stderr = stderr.replace(b'\r\n', b'\n').decode()
    return stdout, stderr, exit_code


@pytest.fixture(scope="function")
def project(request):
    if os.path.exists(PATH):
        # shutil.copytree(src="examples/valid_raw_data", dst=PATH)
        shutil.rmtree(PATH)
    shutil.copytree(src="examples/valid_raw_data", dst=PATH, symlinks=True)

    project = 1

    yield project


def test_validate(project):
    stdout, stderr, exit_code = cli(
        ["child-project", "validate", PATH]
    )
    print(stdout)
    print(stderr)
    assert exit_code == 0


def test_overview(project):
    stdout, stderr, exit_code = cli(
        ["child-project", "overview", PATH]
    )
    assert exit_code == 0
    assert stdout == f"\n\x1b[1m2 recordings with 0.00 hours 2 locally (1 discarded)\x1b[0m:\n\x1b[94mdate range :\x1b[0m 2020-04-20 to 2020-04-21\n\x1b[94mdevices :\x1b[0m usb (0.00h 2/2 locally);\n\n\x1b[1m1 participants\x1b[0m:\n\x1b[94mage range :\x1b[0m 3.6mo to 3.6mo\n\x1b[94mlanguages :\x1b[0m\n\n\x1b[1mannotations:\x1b[0m\nduration    method algo version       date transcr\n    8.0s automated  VTC       1 2024-04-07         \x1b[94mvtc_present\x1b[0m\n\n"
    assert stderr == f"\x1b[33mWARNING \x1b[0m column(s) child_dob overwritten by {Path('output/cli/metadata/children/0_test.csv')} \x1b[35m[ChildProject.projects]\x1b[0m\n\x1b[33mWARNING \x1b[0m column(s) notes overwritten by {Path('output/cli/metadata/recordings/1_very_confidential.csv')} \x1b[35m[ChildProject.projects]\x1b[0m\n\x1b[33mWARNING \x1b[0m column(s) date_iso overwritten by {Path('output/cli/metadata/recordings/0_confidential.csv')} \x1b[35m[ChildProject.projects]\x1b[0m\n"


def test_init():
    shutil.rmtree(PATH, ignore_errors=True)
    stdout, stderr, exit_code = cli(
        ["child-project", "init", PATH]
    )
    assert exit_code == 0


def test_import_annotations(project):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "import-annotations",
            PATH,
            "--annotations",
            "examples/valid_raw_data/annotations/input_short.csv",
        ]
    )
    assert exit_code == 0


def test_import_automated(project):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "automated-import",
            PATH,
            "--set",
            "vtc_rttm",
            "--format",
            "vtc_rttm",
        ]
    )
    assert exit_code == 0


def test_compute_durations(project):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "compute-durations",
            PATH
        ]
    )
    assert exit_code == 0

def test_explain(project):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "explain",
            PATH,
            "notes"
        ]
    )
    assert exit_code == 0

    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "explain",
            PATH,
            "non-existent-variable"
        ]
    )
    assert exit_code == 0

def test_process(project):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "process",
            PATH,
            "basic",
            "standard",
            "--format=wav",
            "--sampling=16000",
            "--codec==pcm_s16le",
        ]
    )
    assert exit_code == 0
    
def test_compare_recordings(project):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "compare-recordings",
            PATH,
            "sound.wav",
            "sound2.wav",
            "--interval",
            "10"
        ]
    )
    assert exit_code == 0

def test_derive_annotations(project):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "derive-annotations",
            PATH,
            "-i",
            "vtc_present",
            "-o",
            "vtc/conversations",
            "conversations",
        ]
    )
    assert exit_code == 0

def test_merge_annotations(project):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "merge-annotations",
            PATH,
            "--left-set",
            "vtc_present",
            "--right-set",
            "vtc_present",
            "--left-columns",
            "speaker_type",
            "--right-columns",
            "speaker_type",
            "--output-set",
            "vtc/self-merge",
        ]
    )
    print(stdout)
    print(stderr)
    assert exit_code == 0

def test_intersect_annotations(project):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "intersect-annotations",
            PATH,
            "--destination",
            str(project.path / "test.csv"),
            "--sets",
            "vtc_present",
            "vtc_present",
        ]
    )
    assert exit_code == 0

def test_remove_annotations(project):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "remove-annotations",
            PATH,
            "--set",
            "vtc_present",
        ]
    )
    assert exit_code == 0

def test_rename_annotations(project):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "rename-annotations",
            PATH,
            "--set",
            "vtc_present",
            "--new-set",
            "vtc_renamed",
        ]
    )
    assert exit_code == 0

def test_sampler(project):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "sampler",
            PATH,
            str(project.path / "test.csv"),
            "periodic"
            "--length",
            "100",
            "--period",
            "400",
            "--offset",
            "200"
        ]
    )
    assert exit_code == 0

# TODO: extend to all zoniverse sub-functions
def test_zooniverse(project):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "zooniverse",
            PATH,
            "-h",
        ]
    )
    assert exit_code == 0

def test_eaf_builder(project):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "eaf-builder",
            "--destination",
            "samples",
            "--segments",
            "tests/truth/segments_metrics.csv"
            "--eaf-type",
            "random",
            "--template",
            "basic",
        ]
    )
    assert exit_code == 0

def test_anonymize(project):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "anonymize",
            PATH,
            "--input-set",
            "vtc_present",
            "--output-set",
            "vtc_anonymized"
        ]
    )
    assert exit_code == 0


def test_metrics(project):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "metrics",
            PATH,
            "test",
            "aclew",
            "--vtc",
            "vtc_present",
        ]
    )
    assert exit_code == 0

def test_metrics_specification(project):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "metrics-specification",
            "tests/data/parameters_metrics.yml",
        ]
    )
    assert exit_code == 0

def test_conversations_summary(project):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "metrics",
            PATH,
            "test",
            "standard",
            "--set",
            "vtc_present",
        ]
    )
    assert exit_code == 0

def test_conversations_specification(project):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "conversations-specification",
            "tests/data/conversations_parameters.yml",
        ]
    )
    assert exit_code == 0

def test_sets_metadata(project):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "sets-metadata",
            PATH,
        ]
    )
    assert exit_code == 0
    assert stdout == f"duration    method algo version       date transcr\n    8000 automated  VTC       1 2024-04-07         \x1b[94mvtc_present\x1b[0m\n\n"
    assert stderr == f"\x1b[33mWARNING \x1b[0m column(s) child_dob overwritten by {Path('output/cli/metadata/children/0_test.csv')} \x1b[35m[ChildProject.projects]\x1b[0m\n\x1b[33mWARNING \x1b[0m column(s) notes overwritten by {Path('output/cli/metadata/recordings/1_very_confidential.csv')} \x1b[35m[ChildProject.projects]\x1b[0m\n\x1b[33mWARNING \x1b[0m column(s) date_iso overwritten by {Path('output/cli/metadata/recordings/0_confidential.csv')} \x1b[35m[ChildProject.projects]\x1b[0m\n"