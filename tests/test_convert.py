from ChildProject.projects import ChildProject
from ChildProject.pipelines.converters import AudioConversionPipeline
import numpy as np
import os
import pandas as pd
import pytest
import shutil

@pytest.fixture(scope='function')
def project(request):
    if not os.path.exists("output/convert"):
            shutil.copytree(src = "examples/valid_raw_data", dst = "output/convert")

    project = ChildProject("output/convert")
    project.read()
    yield project

def test_basic(project):
    converted, parameters = AudioConversionPipeline().run(
        converter = 'basic',
        path = project.path,
        name = 'test',
        format = 'wav',
        codec = 'pcm_s16le',
        sampling = 16000
    )

    recordings = project.recordings
    converted_recordings = pd.read_csv(converted)

    assert np.isclose(4000, project.compute_recordings_duration()['duration'].sum()), "audio duration equals expected value"
    assert os.path.exists(os.path.join(project.path, ChildProject.CONVERTED_RECORDINGS, "test")), "missing converted recordings folder"
    assert recordings.shape[0] == converted_recordings.shape[0], "conversion table is incomplete"
    assert all(converted_recordings['success'].tolist()), "not all recordings were successfully converted"
    assert all([
        os.path.exists(os.path.join(project.path, ChildProject.CONVERTED_RECORDINGS, "test", f))
        for f in converted_recordings['converted_filename'].tolist()
    ]), "recording files are missing"

def test_vetting(project):
    pd.DataFrame([{
        'recording_filename': 'sound.wav',
        'segment_onset': 1000,
        'segment_offset': 3000
    }]).to_csv(os.path.join(project.path, 'segments.csv'))

    AudioConversionPipeline().run(
        converter = 'vetting',
        path = project.path,
        name = 'vetted',
        segments_path = os.path.join(project.path, 'segments.csv')
    )