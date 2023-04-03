import os
import pandas as pd

import pytest
import shutil

from ChildProject.projects import ChildProject
from ChildProject.pipelines.samplers import PeriodicSampler
from ChildProject.pipelines.zooniverse import ZooniversePipeline, pad_interval
from ChildProject.pipelines.fake_panoptes import LOCATION_FAIL


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

BASE_CHUNKS = os.path.join('tests', 'data', 'chunks_test.csv')
# the test is done on a fake Panoptes package to avoid pushing real stuff to zooniverse
# but the real panoptes package may change and produce other errors
@pytest.mark.parametrize("location_fail,amount,ignore_errors,record_orphan,result", 
                         [(False,None,False,False,'stale_obj_stop.csv'),
                         (True,30,False,False,'loc_invalid_stop.csv'),
                         (False,None,True,True,'stale_obj_continue_orphan.csv'),
                         (False,30,True,False,'no_errors.csv'),
                         (True,30,True,True,'loc_invalid_continue.csv'),
                         ])
def test_uploading(location_fail, amount, ignore_errors, record_orphan, result):
    new_path = os.path.join('output', 'zooniverse', 'chunks_test.csv')
    os.makedirs("output/zooniverse", exist_ok=True)
    
    df = pd.read_csv(BASE_CHUNKS)
    
    zooniverse = ZooniversePipeline()
    
    if amount is None: amount = 1000
    
    if not location_fail: df = df[df['mp3'] != LOCATION_FAIL]
    
    df.to_csv(new_path,index=False)
    
    zooniverse.upload_chunks(new_path,
                             404,
                             'test_subject_set',
                             'test_username',
                             'test_password',
                             amount,
                             ignore_errors,
                             record_orphan,
                             test_endpoint=True,
                             )
    
    truth = pd.read_csv(os.path.join('tests','truth','zoochunks',result))
    #shutil.copy(new_path, os.path.join('tests','truth','zoochunks',result))
    
    pd.testing.assert_frame_equal(truth, pd.read_csv(new_path), check_like=True)
        
    
    
def test_link_orphan():
    pass
    
def test_reset_orphan():
    pass
    
def test_classification():
    pass