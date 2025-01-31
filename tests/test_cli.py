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
    assert stdout == b"\n\x1b[1m2 recordings with 0.00 hours 2 locally (0 discarded)\x1b[0m:\n\x1b[94mdate range :\x1b[0m 2020-04-20 to 2020-04-21\n\x1b[94mdevices :\x1b[0m usb (0.00h 2/2 locally);\n\n\x1b[1m1 participants\x1b[0m:\n\x1b[94mage range :\x1b[0m 3.6mo to 3.6mo\n\x1b[94mlanguages :\x1b[0m\n\n\x1b[1mannotations:\x1b[0m\nduration    method algo version       date transcr\n    8.0s automated  VTC       1 2024-04-07         \x1b[94mvtc_present\x1b[0m\n\n"
    assert stderr == b"\x1b[33mWARNING \x1b[0m column(s) child_dob overwritten by output/cli/metadata/children/0_test.csv \x1b[35m[ChildProject.projects]\x1b[0m\n\x1b[33mWARNING \x1b[0m column(s) notes overwritten by output/cli/metadata/recordings/1_very_confidential.csv \x1b[35m[ChildProject.projects]\x1b[0m\n\x1b[33mWARNING \x1b[0m column(s) date_iso overwritten by output/cli/metadata/recordings/0_confidential.csv \x1b[35m[ChildProject.projects]\x1b[0m\n"


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

def test_sets_metadata(project):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "sets-metadata",
            PATH,
        ]
    )
    assert exit_code == 0
    assert stdout == b"duration    method algo version       date transcr\n    8000 automated  VTC       1 2024-04-07         \x1b[94mvtc_present\x1b[0m\n\n"
    assert stderr == b"\x1b[33mWARNING \x1b[0m column(s) child_dob overwritten by output/cli/metadata/children/0_test.csv \x1b[35m[ChildProject.projects]\x1b[0m\n\x1b[33mWARNING \x1b[0m column(s) notes overwritten by output/cli/metadata/recordings/1_very_confidential.csv \x1b[35m[ChildProject.projects]\x1b[0m\n\x1b[33mWARNING \x1b[0m column(s) date_iso overwritten by output/cli/metadata/recordings/0_confidential.csv \x1b[35m[ChildProject.projects]\x1b[0m\n"