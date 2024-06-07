import os
import pandas as pd

import pytest
import shutil

from ChildProject.projects import ChildProject
from ChildProject.pipelines.samplers import PeriodicSampler
from ChildProject.pipelines.zooniverse import ZooniversePipeline, pad_interval
from ChildProject.pipelines.fake_panoptes import LOCATION_FAIL
from ChildProject.pipelines.zooniverse import CHUNKS_DTYPES


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

    chunks = pd.read_csv(chunks, dtype=CHUNKS_DTYPES)

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
                         (True,30,True,True,'max_subjects_continue.csv'),
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
                             test_endpoint=True if result != 'max_subjects_continue.csv' else 2,
                             )
    
    truth = pd.read_csv(os.path.join('tests','truth','zoochunks',result), dtype=CHUNKS_DTYPES)
    # shutil.copy(new_path, os.path.join('tests','truth','zoochunks',result))
    
    pd.testing.assert_frame_equal(truth, pd.read_csv(new_path, dtype=CHUNKS_DTYPES), check_like=True)
   
#might benefit from a test using invalid csv or/and a test with a csv having no orphan chunk     
BASE_ORPHAN_CHUNKS = os.path.join('tests', 'data', 'chunks_test_orphan.csv')   
@pytest.mark.parametrize("ignore_errors,result", 
                         [(False,'link_orphan_stop.csv'),
                         (True,'link_orphan_continue.csv'),
                         ])
def test_link_orphan(ignore_errors, result):
    new_path = os.path.join('output', 'zooniverse', 'chunks_test_orphan.csv')
    os.makedirs("output/zooniverse", exist_ok=True)
    
    shutil.copy(BASE_ORPHAN_CHUNKS,new_path)
    
    zooniverse = ZooniversePipeline()
    
    zooniverse.link_orphan_subjects(new_path,
                                    404,
                                    'test_subject_set',
                                    'test_username',
                                    'test_password',
                                    ignore_errors,
                                    test_endpoint=True,
                                    )
    # shutil.copy(new_path, os.path.join('tests','truth','zoochunks',result))
    pd.testing.assert_frame_equal(pd.read_csv(os.path.join('tests','truth','zoochunks',result), dtype=CHUNKS_DTYPES),
                                  pd.read_csv(new_path, dtype=CHUNKS_DTYPES))
    
@pytest.mark.parametrize("result", 
                         [('reset_orphan.csv'),
                         ])
def test_reset_orphan(result):
    new_path = os.path.join('output', 'zooniverse', 'chunks_test_reset_orphan.csv')
    os.makedirs("output/zooniverse", exist_ok=True)
    
    shutil.copy(BASE_ORPHAN_CHUNKS,new_path)
    
    zooniverse = ZooniversePipeline()
    
    zooniverse.reset_orphan_subjects(new_path)
    
    # shutil.copy(new_path, os.path.join('tests','truth','zoochunks',result))
    pd.testing.assert_frame_equal(pd.read_csv(os.path.join('tests','truth','zoochunks',result), dtype=CHUNKS_DTYPES),
                                  pd.read_csv(new_path, dtype=CHUNKS_DTYPES))
    
def test_classification():
    pass