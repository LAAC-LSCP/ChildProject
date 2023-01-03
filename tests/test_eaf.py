import os
import pandas as pd
from pympi import Eaf

from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from ChildProject.pipelines.samplers import PeriodicSampler
from ChildProject.pipelines.eafbuilder import EafBuilderPipeline

def test_periodic():
    os.makedirs('output/eaf', exist_ok = True)

    project = ChildProject('examples/valid_raw_data')
    project.read()

    sampler = PeriodicSampler(project, 500, 500, 250, recordings = ['sound.wav'])
    sampler.sample()
    sampler.segments.to_csv('output/eaf/segments.csv')

    eaf_builder = EafBuilderPipeline()
    eaf_builder.run(
        destination = 'output/eaf',
        segments = 'output/eaf/segments.csv',
        eaf_type = 'periodic',
        template = 'basic',
        context_onset = 250,
        context_offset = 250
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
        sampler.segments[['segment_onset', 'segment_offset']].sort_values(['segment_onset', 'segment_offset']).reset_index(drop = True)
    )

    assert eaf.media_descriptors[0]['MEDIA_URL'] == 'sound.wav'

def test_prefill():
    os.makedirs('output/eaf', exist_ok = True)

    project = ChildProject('examples/valid_raw_data')
    am = AnnotationManager(project)
    am.read()

    sampler = PeriodicSampler(project, 500, 500, 250, recordings = ['sound.wav'])
    sampler.sample()
    sampler.segments.to_csv('output/eaf/segments.csv')

    eaf_builder = EafBuilderPipeline()
    eaf_builder.run(
        destination = 'output/eaf',
        segments = 'output/eaf/segments_prefill.csv',
        eaf_type = 'periodic',
        template = 'basic',
        context_onset = 250,
        context_offset = 250
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
        sampler.segments[['segment_onset', 'segment_offset']].sort_values(['segment_onset', 'segment_offset']).reset_index(drop = True)
    )

    assert eaf.media_descriptors[0]['MEDIA_URL'] == 'sound.wav'