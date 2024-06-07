from ChildProject.projects import ChildProject, CONVERTED_RECORDINGS
from ChildProject.pipelines.processors import AudioProcessingPipeline
import numpy as np
import os
import pandas as pd
import pytest
import shutil
from pathlib import Path

PATH = Path('output', 'process')
@pytest.fixture(scope="function")
def project(request):
    if os.path.exists(PATH):
        shutil.rmtree(PATH)
    shutil.copytree(src="examples/valid_raw_data", dst=PATH)

    project = ChildProject(PATH)
    project.read()

    yield project


def test_basic(project):
    processed, parameters = AudioProcessingPipeline().run(
        processor="basic",
        path=project.path,
        name="test",
        format="wav",
        codec="pcm_s16le",
        sampling=16000,
    )

    recordings = project.recordings
    converted_recordings = pd.read_csv(processed)

    assert np.isclose(
        8000, project.compute_recordings_duration()["duration"].sum()
    ), "audio duration equals expected value"
    assert os.path.exists(
        os.path.join(project.path, CONVERTED_RECORDINGS, "test")
    ), "missing processed recordings folder"
    assert (
        recordings.shape[0] == converted_recordings.shape[0]
    ), "conversion table is incomplete"
    assert all(
        converted_recordings["success"].tolist()
    ), "not all recordings were successfully processed"
    assert all(
        [
            os.path.exists(
                os.path.join(project.path, CONVERTED_RECORDINGS, "test", f)
            )
            for f in converted_recordings["converted_filename"].tolist()
        ]
    ), "recording files are missing"

def test_standard(project):
    # Starting the audio processing pipeline using the default settings
    processed, parameters = AudioProcessingPipeline().run(
        processor="standard",
        path=project.path,
    )

    recordings = project.recordings
    converted_recordings = pd.read_csv(processed)

    assert np.isclose(
        8000, project.compute_recordings_duration()["duration"].sum()
    ), "audio duration equals expected value"
    assert os.path.exists(
        os.path.join(project.path, CONVERTED_RECORDINGS, "standard")
    ), "missing processed recordings folder"
    assert (
        recordings.shape[0] == converted_recordings.shape[0]
    ), "conversion table is incomplete"
    assert all(
        converted_recordings["success"].tolist()
    ), "not all recordings were successfully processed"
    assert all(
        [
            os.path.exists(
                os.path.join(project.path, CONVERTED_RECORDINGS, "standard", f)
            )
            for f in converted_recordings["converted_filename"].tolist()
        ]
    ), "recording files are missing"


def test_vetting(project):
    pd.DataFrame(
        [
            {
                "recording_filename": "sound.wav",
                "segment_onset": 1000,
                "segment_offset": 3000,
            }
        ]
    ).to_csv(os.path.join(project.path, "segments.csv"))

    AudioProcessingPipeline().run(
        processor="vetting",
        path=project.path,
        name="vetted",
        segments_path=os.path.join(project.path, "segments.csv"),
    )


def test_channel_mapping(project, input=None):
    AudioProcessingPipeline().run(
        processor="channel-mapping",
        path=project.path,
        name="mapping",
        channels=["0,2", "1,0"],
        input_profile=input,
        recordings=["sound.wav"],
    )


def test_custom_input_profile(project):
    test_vetting(project)
    test_channel_mapping(project, input="vetted")

