import subprocess
import pytest
import os
import shutil
from pathlib import Path
from functools import partial
from test_metrics import fake_vocs

import pandas as pd

from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager

from test_conversations import segments

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

    project = ChildProject(PATH)

    yield project


@pytest.fixture(scope="function")
def am(request, project):
    am = AnnotationManager(project)
    # force longer durations to allow for imports
    project.recordings['duration'] = [100000000, 2000000]
    yield am


@pytest.fixture(scope="function")
def dataset_setup(request):
    if os.path.exists(PATH):
        # shutil.copytree(src="examples/valid_raw_data", dst=PATH)
        shutil.rmtree(PATH)
    shutil.copytree(src="examples/valid_raw_data", dst=PATH, symlinks=True)

    yield 1


def test_validate(dataset_setup):
    stdout, stderr, exit_code = cli(
        ["child-project", "validate", PATH]
    )
    assert exit_code == 0


def test_overview(dataset_setup):
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


def test_import_annotations(dataset_setup):
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


def test_import_automated(dataset_setup):
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


def test_compute_durations(dataset_setup):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "compute-durations",
            PATH
        ]
    )
    assert exit_code == 0

def test_explain(dataset_setup):
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

def test_process(dataset_setup):
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
    
def test_compare_recordings(dataset_setup):
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

def test_derive_annotations(dataset_setup):
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

def test_merge_annotations(dataset_setup, project, am):
    input_annotations = pd.read_csv("examples/valid_raw_data/annotations/input.csv")
    input_annotations = input_annotations[
        input_annotations["set"].isin(["vtc_rttm", "alice/output"])
    ]
    am.import_annotations(input_annotations)

    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "merge-annotations",
            PATH,
            "--left-set",
            "vtc_rttm",
            "--right-set",
            "alice/output",
            "--left-columns",
            "speaker_type",
            "--right-columns",
            "words,phonemes",
            "--output-set",
            "alice",
        ]
    )
    assert exit_code == 0

def test_intersect_annotations(dataset_setup):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "intersect-annotations",
            PATH,
            "--destination",
            str(Path(PATH) / "test.csv"),
            "--sets",
            "vtc_present",
            "vtc_present",
        ]
    )
    print(stderr)
    assert exit_code == 0

def test_remove_annotations(dataset_setup):
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

def test_rename_annotations(dataset_setup):
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

def test_sampler(dataset_setup):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "sampler",
            PATH,
            str(Path(PATH) / "test.csv"),
            "periodic",
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
def test_zooniverse(dataset_setup):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "zooniverse",
            "-h",
        ]
    )
    assert exit_code == 0

def test_eaf_builder(dataset_setup):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "eaf-builder",
            "--destination",
            str(Path(PATH) / 'samples'),
            "--segments",
            "tests/truth/segments_metrics.csv",
            "--eaf-type",
            "random",
            "--template",
            "basic",
        ]
    )
    assert exit_code == 0

def test_anonymize(dataset_setup):
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


def test_metrics(dataset_setup):
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "metrics",
            PATH,
            str(Path(PATH) / 'metrics.csv'),
            "aclew",
            "--vtc",
            "vtc_present",
        ]
    )
    assert exit_code == 0

def test_metrics_specification(dataset_setup, project, am):
    data = pd.read_csv("tests/data/lena_its.csv")

    am.import_annotations(
        pd.DataFrame(
            [
                {
                    "set": "specs_its",
                    "raw_filename": "file.its",
                    "time_seek": 0,
                    "recording_filename": "sound.wav",
                    "range_onset": 0,
                    "range_offset": 100000000,
                    "format": "its",
                }
            ]
        ),
        import_function=partial(fake_vocs, data),
    )
    from test_metrics import PATH as met_path
    if os.path.exists(met_path):
        shutil.rmtree(met_path)
    shutil.copytree(src=PATH, dst=met_path, symlinks=True)
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "metrics-specification",
            "tests/data/parameters_metrics.yml",
        ]
    )
    assert exit_code == 0

def test_conversations_summary(dataset_setup, project, am, segments):
    am.import_annotations(
        pd.DataFrame(
            [{"set": "custom_conv",
              "raw_filename": "file.its",
              "time_seek": 0,
              "recording_filename": "sound.wav",
              "range_onset": 0,
              "range_offset": 30000000,
              "format": "csv",
              }]
        ),
        import_function=partial(fake_vocs, segments),
    )

    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "conversations-summary",
            PATH,
            str(Path(PATH) / "test.csv"),
            "--set",
            "custom_conv",
            "standard",
        ]
    )
    assert exit_code == 0

def test_conversations_specification(dataset_setup, project, am, segments):
    am.import_annotations(
        pd.DataFrame(
            [{"set": "custom_conv",
              "raw_filename": "file.its",
              "time_seek": 0,
              "recording_filename": "sound.wav",
              "range_onset": 0,
              "range_offset": 30000000,
              "format": "csv",
              }]
        ),
        import_function=partial(fake_vocs, segments),
    )
    from test_conversations import PATH as conv_path
    if os.path.exists(conv_path):
        shutil.rmtree(conv_path)
    shutil.copytree(src=PATH, dst=conv_path, symlinks=True)
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "conversations-specification",
            "tests/data/conversations_parameters.yml",
        ]
    )
    assert exit_code == 0

def test_sets_metadata(dataset_setup):
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