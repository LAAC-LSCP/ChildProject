from functools import partial
import numpy as np
import os
import pandas as pd
import pytest
import shutil

from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from ChildProject.pipelines.metrics import Metrics, LenaMetrics, AclewMetrics, CustomMetrics, MetricsSpecificationPipeline


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
    try:
        lena = LenaMetrics(project, set="unknown")
    except ValueError as e:
        exception_caught = True

    assert (
        exception_caught == True
    ), "LenaMetrics failed to throw an exception despite an invalid ITS set being provided"
    
    exception_caught = False
    try:
        lm = pd.DataFrame(np.array(
            [["voc_speaker","segments_vtc",'FEM'],         
             ]), columns=["callable","set","speaker"])
        m = Metrics(project, lm,  segments="unknown")
    except ValueError as e:
        exception_caught = True

    assert (
        exception_caught == True
    ), "Metrics failed to throw an exception despite having the segments argument and by having a value different than 'recording_filename'"


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

    lena = LenaMetrics(project, set="lena_its", period='1h', from_time='10:00:00' , to_time= '16:00:00')
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
    
def test_metrics_segments(project):
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
                for set in ["segments_vtc", "segments_alice", "segments_vcm"]
            ]
        ),
        import_function=partial(fake_vocs, data),
    )
    lm = pd.DataFrame(np.array(
            [["voc_speaker","segments_vtc",'FEM'],         
             ["voc_speaker","segments_vtc",'CHI'],
             ["voc_speaker_ph","segments_vtc",'FEM'],         
             ["voc_speaker_ph","segments_vtc",'CHI'],
             ["wc_speaker_ph","segments_alice",'FEM'],
             ["lp_n","segments_vcm",pd.NA],
             ["lp_dur","segments_vcm",pd.NA],
             ]), columns=["callable","set","speaker"])
    metrics = Metrics(project, metrics_list=lm, by="segments", rec_cols='date_iso', child_cols='experiment,child_dob',segments='tests/data/segments.csv')
    metrics.extract()

    truth = pd.read_csv("tests/truth/segments_metrics.csv")

    pd.testing.assert_frame_equal(metrics.metrics, truth, check_like=True)

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

