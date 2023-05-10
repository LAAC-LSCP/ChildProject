import os
import pandas as pd

from ChildProject.projects import ChildProject
from ChildProject.pipelines.samplers import PeriodicSampler
from ChildProject.pipelines.zooniverse import ZooniversePipeline, pad_interval


def test_padding():
    assert pad_interval(300, 800, 500, 1) == (300, 800)
    assert pad_interval(200, 800, 500, 2) == (0, 1000)

    assert pad_interval(300, 900, 500, 1) == (100, 1100)

    assert pad_interval(2000, 2500, 100, 10) == (1750, 2750)

    assert pad_interval(100, 300, 500, 1) == (-50, 450)


def test_extraction():
    os.makedirs("output/zooniverse", exist_ok=True)

    project = ChildProject("examples/valid_raw_data")
    project.read()
    
    project.recordings['duration'] = 4000 #force duration 4s (it is the actual length) to allow for segmenting (otherwise fails)

    sampler = PeriodicSampler(project, 500, 500, 250)
    segments = sampler.sample()
    sampler.segments.to_csv("output/zooniverse/sampled_segments.csv")

    zooniverse = ZooniversePipeline()

    chunks, parameters = zooniverse.extract_chunks(
        path=project.path,
        destination="output/zooniverse",
        keyword="test",
        segments="output/zooniverse/sampled_segments.csv",
        chunks_length=250,
        chunks_min_amount=2,
        spectrogram=True,
    )

    chunks = pd.read_csv(chunks)

    assert len(chunks) == 2 * len(segments)
    assert all(
        chunks["wav"]
        .apply(lambda f: os.path.exists(os.path.join("output/zooniverse/chunks", f)))
        .tolist()
    )
    assert all(
        chunks["mp3"]
        .apply(lambda f: os.path.exists(os.path.join("output/zooniverse/chunks", f)))
        .tolist()
    )
    assert all(
        chunks["png"]
        .apply(lambda f: os.path.exists(os.path.join("output/zooniverse/chunks", f)))
        .tolist()
    )

