import numpy as np
import os
import pandas as pd
import pytest
import shutil

from ChildProject.projects import ChildProject
from ChildProject.pipelines.samplers import (
    PeriodicSampler,
    RandomVocalizationSampler,
    EnergyDetectionSampler,
    HighVolubilitySampler,
    SamplerPipeline
)

@pytest.fixture(scope='function')
def project(request):
    if not os.path.exists("output/samplers"):
        shutil.copytree(src = "examples/valid_raw_data", dst = "output/samplers")

    project = ChildProject("output/samplers")
    project.read()
    yield project

def test_periodic(project):
    project = ChildProject('output/samplers')
    project.read()
    
    project.recordings = project.recordings.merge(
        project.compute_recordings_duration(),
        left_on = 'recording_filename',
        right_on = 'recording_filename'
    )

    sampler = PeriodicSampler(
        project = project,
        length = 1000,
        period = 1000,
        recordings = ['sound.wav']
    )
    sampler.sample()

    duration = project.recordings[project.recordings['recording_filename'] == 'sound.wav']['duration'].iloc[0]

    assert len(sampler.segments) == int(duration/(1000+1000))

def test_energy_detection(project):
    project = ChildProject('output/samplers')
    project.read()

    sampler = EnergyDetectionSampler(
        project = project,
        windows_length = 100,
        windows_spacing = 100,
        windows_count = 1,
        threshold = 0.95,
        low_freq = 200,
        high_freq = 1000,
        recordings = ['sound.wav'],
        threads = 1
    )
    sampler.sample()

    np.testing.assert_array_equal(
        sampler.segments[['segment_onset', 'segment_offset']].values,
        [[1800, 1900]]
    )
    

