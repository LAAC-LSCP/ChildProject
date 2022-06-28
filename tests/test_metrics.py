from functools import partial
import numpy as np
import os
import pandas as pd
import pytest
import shutil

from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from ChildProject.pipelines.metrics import LenaMetrics, AclewMetrics, CustomMetrics, MetricsSpecificationPipeline


def fake_vocs(data, filename):
    return data


@pytest.fixture(scope="function")
def project(request):
    if not os.path.exists("output/metrics"):
        shutil.copytree(src="examples/valid_raw_data", dst="output/metrics")

    project = ChildProject("output/metrics")
    project.read()

    yield project


def test_failures(project):
    exception_caught = False
    try:
        aclew = AclewMetrics(project, vtc="unknown")
    except ValueError as e:
        exception_caught = True

    assert (
        exception_caught == True
    ), "AclewMetrics failed to throw an exception despite an invalid VTC set being provided"
    exception_caught = False

    exception_caught = False
    try:
        lena = LenaMetrics(project, set="unknown")
    except ValueError as e:
        exception_caught = True

    assert (
        exception_caught == True
    ), "LenaMetrics failed to throw an exception despite an invalid ITS set being provided"
    exception_caught = False


def test_aclew(project):
    data = pd.read_csv("tests/data/aclew.csv")

    am = AnnotationManager(project)
    am.import_annotations(
        pd.DataFrame(
            [
                {
                    "set": set,
                    "raw_filename": "file.rttm",
                    "time_seek": 0,
                    "recording_filename": "sound.wav",
                    "range_onset": 0,
                    "range_offset": 4000,
                    "format": "rttm",
                }
                for set in ["aclew_vtc", "aclew_alice", "aclew_vcm"]
            ]
        ),
        import_function=partial(fake_vocs, data),
    )

    aclew = AclewMetrics(project, by="child_id", rec_cols='date_iso', child_cols='experiment,child_dob',vtc='aclew_vtc',alice='aclew_alice',vcm='aclew_vcm')
    aclew.extract()

    truth = pd.read_csv("tests/truth/aclew_metrics.csv")

    pd.testing.assert_frame_equal(aclew.metrics, truth, check_like=True)

def test_lena(project):
    data = pd.read_csv("tests/data/lena_its.csv")

    am = AnnotationManager(project)
    am.import_annotations(
        pd.DataFrame(
            [
                {
                    "set": "lena_its",
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

    lena = LenaMetrics(project, set="lena_its", period='1h', from_time='10:00' , to_time= '16:00')
    lena.extract()

    truth = pd.read_csv("tests/truth/lena_metrics.csv")

    pd.testing.assert_frame_equal(lena.metrics, truth, check_like=True)

def test_custom(project):
    am = AnnotationManager(project)
    
    data = pd.read_csv("tests/data/lena_its.csv")

    am.import_annotations(
        pd.DataFrame(
            [
                {
                    "set": "custom_its",
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
        
    parameters="tests/data/list_metrics.csv"
    cmm = CustomMetrics(project, parameters)
    cmm.extract()

    truth = pd.read_csv("tests/truth/custom_metrics.csv")

    pd.testing.assert_frame_equal(cmm.metrics, truth, check_like=True)

def test_specs(project):
    data = pd.read_csv("tests/data/lena_its.csv")
    
    am = AnnotationManager(project)
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
        
    msp = MetricsSpecificationPipeline()
    
    parameters = "tests/data/parameters_metrics.yml"
    msp.run(parameters)
    
    output = pd.read_csv(msp.destination)
    truth = pd.read_csv("tests/truth/specs_metrics.csv")

    pd.testing.assert_frame_equal(output, truth, check_like=True)
    
    new_params = msp.parameters_path
    msp.run(new_params)
    
    output = pd.read_csv(msp.destination)
    
    pd.testing.assert_frame_equal(output, truth, check_like=True)

