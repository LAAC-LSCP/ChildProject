import numpy as np
import os
import pandas as pd
import pytest
import shutil

from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from ChildProject.pipelines.metrics import LenaMetrics, AclewMetrics

def fake_vocs(filename):
    return pd.read_csv('tests/data/aclew.csv')

@pytest.fixture(scope='function')
def project(request):
    if not os.path.exists("output/metrics"):
        shutil.copytree(src = "examples/valid_raw_data", dst = "output/metrics")

    project = ChildProject("output/metrics")
    project.read()

    yield project

def test_aclew(project):    
    am = AnnotationManager(project)
    am.import_annotations(pd.DataFrame([
        {
            'set': set,
            'raw_filename': 'file.rttm',
            'time_seek': 0,
            'recording_filename': 'sound.wav',
            'range_onset': 0,
            'range_offset': 4000,
            'format': 'rttm'
        }
        for set in ['vtc', 'alice', 'vcm']
    ]), import_function = fake_vocs)

    aclew = AclewMetrics(project, by = 'child_id')
    aclew.extract()
    
    truth = pd.read_csv('tests/truth/aclew_metrics.csv', index_col = 'child_id')

    pd.testing.assert_frame_equal(
        aclew.metrics,
        truth
    )