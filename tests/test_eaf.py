from functools import partial
import os
import pandas as pd
from pympi import Eaf
import shutil
import pytest

from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from ChildProject.pipelines.samplers import PeriodicSampler
from ChildProject.pipelines.eafbuilder import EafBuilderPipeline

IMP_FROM = 'vtc'

def fake_vocs(data, filename):
    return data

@pytest.fixture(scope="function")
def project(request):
    if not os.path.exists("output/eaf"):
        shutil.copytree(src="examples/valid_raw_data", dst="output/eaf")

    project = ChildProject("output/eaf")
    project.read()

    yield project

def test_periodic(project):
    """
    os.makedirs('output/eaf', exist_ok = True)

    project = ChildProject('examples/valid_raw_data')
    project.read()
    
    am = AnnotationManager(project)
    am.read()
    """
    data = pd.read_csv("tests/data/eaf_segments.csv")
    
    am = AnnotationManager(project)
    am.import_annotations(
        pd.DataFrame(
            [
                {
                    "set": "vtc",
                    "raw_filename": "file.rttm",
                    "time_seek": 0,
                    "recording_filename": "sound.wav",
                    "range_onset": 0,
                    "range_offset": 4000,
                    "format": "vtc_rttm",
                }
            ]
        ),
        import_function=partial(fake_vocs, data),
    )
        
    sampler = PeriodicSampler(project, 500, 500, 250, recordings = ['sound.wav'])
    sampler.sample()
    sampler.segments.to_csv('output/eaf/segments.csv')
    
    ranges = sampler.segments.rename(
                    columns={
                        "segment_onset": "range_onset",
                        "segment_offset": "range_offset",
                    }
                )
    annotations = am.get_within_ranges(ranges, [IMP_FROM], 'warn')
    #annotations = am.annotations[am.annotations["set"] == IMP_FROM].drop_duplicates(['set', 'recording_filename', 'time_seek', 'range_onset', 'range_offset', 'raw_filename', 'format', 'filter'],ignore_index=True)
    annot_segments = am.get_segments(annotations)

    eaf_builder = EafBuilderPipeline()
    eaf_builder.run(
        destination = 'output/eaf',
        segments = 'output/eaf/segments.csv',
        eaf_type = 'periodic',
        template = 'basic',
        context_onset = 250,
        context_offset = 250,
        path='output/eaf',
        import_speech_from='vtc',
    )

    eaf = Eaf('output/eaf/sound/sound_periodic_basic.eaf')

    code = eaf.tiers['code_periodic'][0]
    segments = []

    for pid in code:
        (start_ts, end_ts, value, svg_ref) = code[pid]
        (start_t, end_t) = (eaf.timeslots[start_ts], eaf.timeslots[end_ts])
        segments.append({'segment_onset': int(start_t), 'segment_offset': int(end_t)})

    segments = pd.DataFrame(segments)
    
    

    pd.testing.assert_frame_equal(
        segments[['segment_onset', 'segment_offset']].sort_values(['segment_onset', 'segment_offset']).reset_index(drop = True),
        sampler.segments[['segment_onset', 'segment_offset']].sort_values(['segment_onset', 'segment_offset']).reset_index(drop = True),
        check_dtype=False,
    )
    
    segments = []
    vtc_speech = eaf.tiers['VTC-SPEECH'][0]  
    for pid in vtc_speech:
        (start_ts, end_ts, value, svg_ref) = vtc_speech[pid]
        (start_t, end_t) = (eaf.timeslots[start_ts], eaf.timeslots[end_ts])
        segments.append({'segment_onset': int(start_t), 'segment_offset': int(end_t)})
        
    segments = pd.DataFrame(segments)
    
    speech_segs = annot_segments[pd.isnull(annot_segments['speaker_type'])]
    
    pd.testing.assert_frame_equal(
        segments[['segment_onset', 'segment_offset']].sort_values(['segment_onset', 'segment_offset']).reset_index(drop = True),
        speech_segs[['segment_onset', 'segment_offset']].sort_values(['segment_onset', 'segment_offset']).reset_index(drop = True)
    )
        
    segments = []
    vtc_chi = eaf.tiers['VTC-CHI'][0]
    for pid in vtc_chi:
        (start_ts, end_ts, value, svg_ref) = vtc_chi[pid]
        (start_t, end_t) = (eaf.timeslots[start_ts], eaf.timeslots[end_ts])
        segments.append({'segment_onset': int(start_t), 'segment_offset': int(end_t)})
        
    segments = pd.DataFrame(segments)
    
    chi_segs = annot_segments[annot_segments['speaker_type'] == 'CHI']
    
    pd.testing.assert_frame_equal(
        segments[['segment_onset', 'segment_offset']].sort_values(['segment_onset', 'segment_offset']).reset_index(drop = True),
        chi_segs[['segment_onset', 'segment_offset']].sort_values(['segment_onset', 'segment_offset']).reset_index(drop = True)
    )
    
    segments = []
    vtc_och = eaf.tiers['VTC-OCH'][0]
    for pid in vtc_och:
        (start_ts, end_ts, value, svg_ref) = vtc_och[pid]
        (start_t, end_t) = (eaf.timeslots[start_ts], eaf.timeslots[end_ts])
        segments.append({'segment_onset': int(start_t), 'segment_offset': int(end_t)})
        
    segments = pd.DataFrame(segments)
    
    och_segs = annot_segments[annot_segments['speaker_type'] == 'OCH']
    
    pd.testing.assert_frame_equal(
        segments[['segment_onset', 'segment_offset']].sort_values(['segment_onset', 'segment_offset']).reset_index(drop = True),
        och_segs[['segment_onset', 'segment_offset']].sort_values(['segment_onset', 'segment_offset']).reset_index(drop = True)
    )
    
    segments = []
    vtc_fem = eaf.tiers['VTC-FEM'][0]
    for pid in vtc_fem:
        (start_ts, end_ts, value, svg_ref) = vtc_fem[pid]
        (start_t, end_t) = (eaf.timeslots[start_ts], eaf.timeslots[end_ts])
        segments.append({'segment_onset': int(start_t), 'segment_offset': int(end_t)})
        
    segments = pd.DataFrame(segments)
    
    fem_segs = annot_segments[annot_segments['speaker_type'] == 'FEM']
    
    pd.testing.assert_frame_equal(
        segments[['segment_onset', 'segment_offset']].sort_values(['segment_onset', 'segment_offset']).reset_index(drop = True),
        fem_segs[['segment_onset', 'segment_offset']].sort_values(['segment_onset', 'segment_offset']).reset_index(drop = True)
    )

    assert eaf.media_descriptors[0]['MEDIA_URL'] == 'sound.wav'
