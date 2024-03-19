from functools import partial
import os
import pandas as pd
from pympi import Eaf
import shutil
import pytest


from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from ChildProject.pipelines.samplers import PeriodicSampler
from ChildProject.pipelines.eafbuilder import EafBuilderPipeline, create_eaf

IMP_FROM = 'vtc'
PATH = os.path.join('output', 'eaf')

def fake_vocs(data, filename):
    return data

@pytest.fixture(scope="function")
def project(request):
    if os.path.exists(PATH):
        shutil.rmtree(PATH)
    shutil.copytree(src="examples/valid_raw_data", dst=PATH)

    project = ChildProject(PATH)
    project.read()

    yield project


IMP = pd.DataFrame({'segment_onset': [10], 'segment_offset': [15], 'speaker_type': ['FEM']})
TEMP = os.path.join('ChildProject', 'templates', 'basic.etf')
@pytest.mark.parametrize(("etf_path,output_dir,recording_filename,timestamps_list,eaf_type,context_on,context_off,speech_segments,imported_set,error"),
[[5, PATH, 'sound.wav', [], 'periodic', 0, 0, IMP, 'vtc', FileNotFoundError],
['README.md', PATH, 'sound.wav', [], 'periodic', 0, 0, IMP, 'vtc', Exception],
[TEMP, 6, 'sound.wav', [], 'periodic', 0, 0, IMP, 'vtc', TypeError],
[TEMP, PATH, 8, [], 'periodic', 0, 0, IMP, 'vtc', TypeError],
[TEMP, PATH, 'sound.wav', 5, 'periodic', 0, 0, IMP, 'vtc', TypeError],
[TEMP, PATH, 'sound.wav', [(5, 'abc')], 'periodic', 0, 0, IMP, 'vtc', ValueError],
[TEMP, PATH, 'sound.wav', [(5, 10)], 'periodic', 'xp', 0, IMP, 'vtc', TypeError],
[TEMP, PATH, 'sound.wav', [(5, 10)], 'periodic', 0, 0, 'x', 'vtc', AttributeError],
[TEMP, PATH, 'sound.wav', [(5, 10)], 'periodic', 0, 0, IMP.drop(columns=['segment_offset']), 'vtc', KeyError],
[TEMP, PATH, 'sound.wav', [(5, 10)], 'periodic', 0, 0, IMP, 5, AttributeError],
    ])
def test_create_eaf_inputs(project, etf_path, output_dir, recording_filename, timestamps_list, eaf_type, context_on,
                           context_off, speech_segments, imported_set, error):
    with pytest.raises(error):
        create_eaf(etf_path, 'sound', output_dir, recording_filename, timestamps_list, eaf_type, context_on, context_off,
                   speech_segments, imported_set, 'vtc_rttm')

def test_create_eaf(project):

    timestamps_list = [(10, 20), (30, 40), (50, 60)]

    create_eaf(TEMP, 'sound', os.path.join(PATH, 'extra/eaf'), 'sound.wav', timestamps_list, 'periodic', 10, 10,
               IMP, 'vtc', 'vtc_rttm')

    eaf = Eaf(os.path.join(PATH, 'extra/eaf/sound.eaf'))

    code = eaf.tiers['code_periodic'][0]
    segments = []

    for pid in code:
        (start_ts, end_ts, value, svg_ref) = code[pid]
        (start_t, end_t) = (eaf.timeslots[start_ts], eaf.timeslots[end_ts])
        segments.append({'segment_onset': int(start_t), 'segment_offset': int(end_t)})

    timestamps = []
    for pid in timestamps_list:
        timestamps.append({'segment_onset': pid[0], 'segment_offset': pid[1]})

    segments = pd.DataFrame(segments)
    timestamps = pd.DataFrame(timestamps)

    pd.testing.assert_frame_equal(
        segments[['segment_onset', 'segment_offset']].sort_values(['segment_onset', 'segment_offset']).reset_index(
            drop=True),
        timestamps[['segment_onset', 'segment_offset']].sort_values(
            ['segment_onset', 'segment_offset']).reset_index(drop=True),
        check_dtype=False,
    )

    segments = []
    vtc_speech = eaf.tiers['VTC-FEM'][0]
    for pid in vtc_speech:
        (start_ts, end_ts, value, svg_ref) = vtc_speech[pid]
        (start_t, end_t) = (eaf.timeslots[start_ts], eaf.timeslots[end_ts])
        segments.append({'segment_onset': int(start_t), 'segment_offset': int(end_t)})

    segments = pd.DataFrame(segments)

    pd.testing.assert_frame_equal(
        segments[['segment_onset', 'segment_offset']].sort_values(['segment_onset', 'segment_offset']).reset_index(
            drop=True),
        IMP[['segment_onset', 'segment_offset']].sort_values(['segment_onset', 'segment_offset']).reset_index(
            drop=True)
    )

    assert eaf.media_descriptors[0]['MEDIA_URL'] == 'sound.wav'


# @pytest.mark.parametrize("segments,type,template,context_onset,context_offset,path,import_speech_from",
#                          [])
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
        
    sampler = PeriodicSampler(project, 500, 500, 250, recordings=['sound.wav'])
    sampler.sample()
    sampler.segments.to_csv(os.path.join(PATH, 'segments.csv'))
    
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
        destination=os.path.join(PATH, 'extra', 'eaf'),
        segments=os.path.join(PATH, 'segments.csv'),
        eaf_type='periodic',
        template='basic',
        context_onset=250,
        context_offset=250,
        path=PATH,
        import_speech_from='vtc',
    )

    eaf = Eaf(os.path.join(PATH, 'extra/eaf/sound_periodic_basic.eaf'))

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
