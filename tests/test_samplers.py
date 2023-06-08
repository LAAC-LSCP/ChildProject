import os
import pandas as pd
import pytest
import shutil
from functools import partial

from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from ChildProject.pipelines.samplers import (
    PeriodicSampler,
    RandomVocalizationSampler,
    EnergyDetectionSampler,
    HighVolubilitySampler,
    ConversationSampler,
    SamplerPipeline,
)


def fake_conversation(data, filename):
    return data


@pytest.fixture(scope="function")
def project(request):
    if not os.path.exists("output/samplers"):
        shutil.copytree(src="examples/valid_raw_data", dst="output/samplers")

    project = ChildProject("output/samplers")
    project.read()
    yield project


def test_periodic(project):
    sampler = PeriodicSampler(
        project=project, length=1000, period=1000, recordings=["sound.wav"]
    )
    sampler.sample()

    duration = project.recordings[
        project.recordings["recording_filename"] == "sound.wav"
    ]["duration"].iloc[0]

    assert len(sampler.segments) == int(duration / (1000 + 1000))


def test_energy_detection(project):
    sampler = EnergyDetectionSampler(
        project=project,
        windows_length=100,
        windows_spacing=100,
        windows_count=1,
        threshold=0.98,
        low_freq=200,
        high_freq=1000,
        threads=1,
        by="recording_filename",
    )
    sampler.sample()

    pd.testing.assert_frame_equal(
        sampler.segments[["segment_onset", "segment_offset"]],
        pd.DataFrame(
            [[1900, 2000], [1900, 2000]], columns=["segment_onset", "segment_offset"]
        ),
        check_dtype=False,
    )

    # group by child_id
    sampler = EnergyDetectionSampler(
        project=project,
        windows_length=100,
        windows_spacing=100,
        windows_count=1,
        threshold=0.99,
        low_freq=200,
        high_freq=1000,
        by="child_id",
        threads=1,
    )
    sampler.sample()

    pd.testing.assert_frame_equal(
        sampler.segments[["recording_filename", "segment_onset", "segment_offset"]],
        pd.DataFrame(
            [["sound.wav", 1900, 2000]],
            columns=["recording_filename", "segment_onset", "segment_offset"],
        ),
        check_dtype=False,
    )


def test_conversation_sampler(project):
    conversations = [
        {"onset": 0, "vocs": 5},
        {"onset": 60 * 1000, "vocs": 10},
        {"onset": 1800 * 1000, "vocs": 15},
    ]
    segments = []
    for conversation in conversations:
        segments += [
            {
                "segment_onset": conversation["onset"] + i * (2000 + 500),
                "segment_offset": conversation["onset"] + i * (2000 + 500) + 2000,
                "speaker_type": ["FEM", "CHI"][i % 2],
            }
            for i in range(conversation["vocs"])
        ]
    segments = pd.DataFrame(segments)

    am = AnnotationManager(project)
    project.recordings['duration'] = 3600 * 1000 * 1000 #forcefully extend the audio duration to accept import here
    am.import_annotations(
        pd.DataFrame(
            [
                {
                    "set": "conv",
                    "raw_filename": "file.rttm",
                    "time_seek": 0,
                    "recording_filename": "sound.wav",
                    "range_onset": 0,
                    "range_offset": 3600 * 1000 * 1000,
                    "format": "rttm",
                }
            ]
        ),
        import_function=partial(fake_conversation, segments),
    )
    sampler = ConversationSampler(
        project, "conv", count=5, interval=1000, speakers=["FEM", "CHI"],
    )
    sampler.sample()
    
    assert len(sampler.segments) == len(conversations)
    assert sampler.segments["segment_onset"].tolist() == [
        conv["onset"]
        for conv in sorted(conversations, key=lambda c: c["vocs"], reverse=True)
    ]

def test_random_vocalization(project):
    segments = [
        {
            'segment_onset': 1000,
            'segment_offset': 2000,
            'speaker_type': speaker
        }
        for speaker in ['CHI', 'FEM', 'MAL']
    ] 

    segments = pd.DataFrame(segments)

    am = AnnotationManager(project)
    am.import_annotations(
        pd.DataFrame(
            [
                {
                    "set": "random",
                    "raw_filename": "file.rttm",
                    "time_seek": 0,
                    "recording_filename": "sound.wav",
                    "range_onset": 0,
                    "range_offset": 4000,
                    "format": "rttm",
                }
            ]
        ),
        import_function=partial(fake_conversation, segments),
    )

    sampler = RandomVocalizationSampler(
        project=project,
        annotation_set="random",
        target_speaker_type=["CHI"],
        sample_size=1,
        threads=1
    )
    sampler.sample()
    
    chi_segments = segments[segments["speaker_type"] == "CHI"]
    pd.testing.assert_frame_equal(
        sampler.segments[["segment_onset", "segment_offset"]].astype(int),
        chi_segments[["segment_onset", "segment_offset"]].astype(int)
    )


def test_exclusion(project):
    excluded = pd.DataFrame(
        [["sound.wav", 250, 750], ["sound.wav", 2000, 4000]],
        columns=["recording_filename", "segment_onset", "segment_offset"],
    )

    sampler = PeriodicSampler(
        project=project, length=1000, period=1000, exclude=excluded
    )
    segments = sampler.sample().sort_values(
        ["recording_filename", "segment_onset", "segment_offset"]
    )

    pd.testing.assert_frame_equal(
        segments[segments["recording_filename"] == "sound.wav"],
        pd.DataFrame(
            [["sound.wav", 0, 250], ["sound.wav", 750, 1000]],
            columns=["recording_filename", "segment_onset", "segment_offset"],
        ),
    )
    pd.testing.assert_frame_equal(
        segments[segments["recording_filename"] == "sound2.wav"],
        pd.DataFrame(
            [["sound2.wav", 0, 1000], ["sound2.wav", 2000, 3000]],
            columns=["recording_filename", "segment_onset", "segment_offset"],
        ),
    )

