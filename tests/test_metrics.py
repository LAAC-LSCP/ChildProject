from functools import partial
import numpy as np
import os
import pandas as pd
import pytest
import shutil
from pathlib import Path

from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from ChildProject.pipelines.metrics import Metrics, LenaMetrics, AclewMetrics, CustomMetrics, MetricsSpecificationPipeline

from ChildProject.pipelines.metricsFunctions import metricFunction, RESERVED

PATH = Path('output/metrics')

def fake_vocs(data, filename):
    return data


@pytest.fixture(scope="function")
def project(request):
    if os.path.exists(PATH):
        # shutil.copytree(src="examples/valid_raw_data", dst="output/annotations")
        shutil.rmtree(PATH)
    shutil.copytree(src="examples/valid_raw_data", dst=PATH)

    project = ChildProject(PATH)
    project.read()

    yield project
    
@pytest.fixture(scope="function")
def am(request, project):
    am= AnnotationManager(project)
    project.recordings['duration'] = [100000000, 2000000] #force longer durations to allow for imports
    yield am    
    
#decorating functions with reserved kwargs should fail
@pytest.mark.parametrize("error", [ValueError, ])
def test_decorator(error):
    for reserved in RESERVED:
        
        with pytest.raises(error):
            
            @metricFunction({reserved},{})
            def fake_function(annotations, duration, **kwargs):
                return 0

def test_failures(project):
    exception_caught = False
    try:
        aclew = AclewMetrics(project, vtc="unknown")
    except ValueError as e:
        exception_caught = True

    assert (
        exception_caught is True
    ), "AclewMetrics failed to throw an exception despite an invalid VTC set being provided"

    exception_caught = False
    try:
        lena = LenaMetrics(project, set="unknown")
    except ValueError as e:
        exception_caught = True

    assert (
        exception_caught is True
    ), "LenaMetrics failed to throw an exception despite an invalid ITS set being provided"
    
    exception_caught = False
    try:
        lm = pd.DataFrame(np.array(
            [["voc_speaker", "vtc_present", 'FEM'],
             ]), columns=["callable", "set", "speaker"])
        Metrics(project, lm,  segments="unknown")
    except AssertionError:
        exception_caught = True

    assert (
        exception_caught is True
    ), "Metrics failed to throw an exception despite having the segments argument and by having a value different than 'recording_filename'"
           
@pytest.mark.parametrize("error,col_change,new_value",
                         [(ValueError, 'name', 'voc_mal_ph_its'),
                          (ValueError, 'name', 'voc_fem_ph_its'),
                          (ValueError, 'speaker', 'FEM'),
                          (None,None,None),
                          ])
def test_metrics(project, am, error, col_change, new_value):
    
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
        
    parameters=pd.read_csv("tests/data/list_metrics.csv")
    
    if error:
        with pytest.raises(error):
            parameters.iloc[0,parameters.columns.get_loc(col_change)] = new_value
            mm = Metrics(project, parameters)
    else:       
        mm = Metrics(project, parameters)
        mm.extract()
    
        truth = pd.read_csv("tests/truth/custom_metrics.csv")
    
        pd.testing.assert_frame_equal(mm.metrics, truth, check_like=True)


def test_aclew(project, am):
    data = pd.read_csv("tests/data/aclew.csv")

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

def test_lena(project, am):
    data = pd.read_csv("tests/data/lena_its.csv")

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

def test_custom(project, am):
    
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
    
def test_metrics_segments(project, am):
    data = pd.read_csv("tests/data/aclew.csv")

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

def test_specs(project, am):
    data = pd.read_csv("tests/data/lena_its.csv")
    
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

def test_metrics_peaks(project, am):
    data = pd.read_csv("tests/data/aclew.csv")

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
                for set in ["peak_vtc", "peak_alice", "peak_vcm"]
            ]
        ),
        import_function=partial(fake_vocs, data),
    )
    lm = pd.DataFrame(np.array(
            [["voc_speaker", "peak_vtc", 'FEM', pd.NA],
             ["voc_speaker", "peak_vtc", 'CHI', pd.NA],
             ["peak_voc_speaker", "peak_vtc", 'FEM', 1000],
             ["peak_voc_speaker", "peak_vtc", 'CHI', 3600000],
             ["voc_speaker_ph", "peak_vtc", 'FEM', pd.NA],
             ["voc_speaker_ph", "peak_vtc", 'CHI', pd.NA],
             ["wc_adu_ph", "peak_alice", pd.NA, pd.NA],
             ["peak_wc_adu", "peak_alice", pd.NA, 1000],
             ["simple_CTC", "peak_vtc", pd.NA, pd.NA],
             ["peak_simple_CTC", "peak_vtc", pd.NA, 1000],
             ]), columns=["callable", "set", "speaker", "period_time"])
    metrics = Metrics(project, metrics_list=lm)
    metrics.extract()

    truth = pd.read_csv("tests/truth/peak_metrics.csv")

    #metrics.metrics.to_csv("tests/truth/peak_metrics.csv",index=False)

    pd.testing.assert_frame_equal(metrics.metrics, truth, check_like=True)