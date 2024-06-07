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
    shutil.copytree(src="examples/valid_raw_data", dst=PATH)

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