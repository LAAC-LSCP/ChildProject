from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from ChildProject.tables import IndexTable
import glob
import pandas as pd
import numpy as np
import os
import pytest
import shutil
import subprocess
import sys

@pytest.fixture(scope='function')
def project(request):
    if not os.path.exists("output/annotations"):
        shutil.copytree(src = "examples/valid_raw_data", dst = "output/annotations")

    project = ChildProject("output/annotations")
    yield project
    
    os.remove("output/annotations/metadata/annotations.csv")
    for raw_annotation in glob.glob("output/annotations/annotations/*.*/converted"):
        shutil.rmtree(raw_annotation)

def test_import(project):
    am = AnnotationManager(project)

    input_annotations = pd.read_csv('examples/valid_raw_data/annotations/input.csv')
    am.import_annotations(input_annotations)
    am.read()
    
    assert am.annotations.shape[0] == input_annotations.shape[0], "imported annotations length does not match input"

    assert all([
        os.path.exists(os.path.join(project.path, 'annotations', f))
        for f in am.annotations['annotation_filename'].tolist()
    ]), "some annotations are missing"

    errors, warnings = am.validate()
    assert len(errors) == 0 and len(warnings) == 0, "malformed annotations detected"

    for dataset in ['eaf', 'textgrid', 'eaf_solis']:
        annotations = am.annotations[am.annotations['set'] == dataset]
        segments = am.get_segments(annotations)
        segments.drop(columns = annotations.columns, inplace = True)

        pd.testing.assert_frame_equal(
            segments.sort_index(axis = 1).sort_values(segments.columns.tolist()).reset_index(drop = True),
            pd.read_csv('tests/truth/{}.csv'.format(dataset)).sort_index(axis = 1).sort_values(segments.columns.tolist()).reset_index(drop = True),
            check_less_precise = True
        )

def test_intersect(project):
    am = AnnotationManager(project)

    input_annotations = pd.read_csv('examples/valid_raw_data/annotations/intersect.csv')
    am.import_annotations(input_annotations)

    a, b = am.intersection(
        am.annotations[am.annotations['set'] == 'textgrid'],
        am.annotations[am.annotations['set'] == 'vtc_rttm']
    )
    
    pd.testing.assert_frame_equal(
        a.sort_index(axis = 1).sort_values(a.columns.tolist()).reset_index(drop = True).drop(columns=['imported_at']),
        pd.read_csv('tests/truth/intersect_a.csv').sort_index(axis = 1).sort_values(a.columns.tolist()).reset_index(drop = True).drop(columns=['imported_at'])
    )

    pd.testing.assert_frame_equal(
        b.sort_index(axis = 1).sort_values(b.columns.tolist()).reset_index(drop = True).drop(columns=['imported_at']),
        pd.read_csv('tests/truth/intersect_b.csv').sort_index(axis = 1).sort_values(b.columns.tolist()).reset_index(drop = True).drop(columns=['imported_at'])
    )

def test_merge(project):
    am = AnnotationManager(project)

    input_annotations = pd.read_csv('examples/valid_raw_data/annotations/input.csv')
    input_annotations = input_annotations[input_annotations['set'].isin(['vtc_rttm', 'alice'])]
    am.import_annotations(input_annotations)
    am.read()

    am.read()
    am.merge_sets(
        left_set = 'vtc_rttm',
        right_set = 'alice',
        left_columns = ['speaker_id','ling_type','speaker_type','vcm_type','lex_type','mwu_type','addresseee','transcription'],
        right_columns = ['phonemes','syllables','words'],
        output_set = 'alice_vtc'
    )
    am.read()

    segments = am.get_segments(am.annotations[am.annotations['set'] == 'alice_vtc'])
    assert segments.shape == am.get_segments(am.annotations[am.annotations['set'] == 'vtc_rttm']).shape

    adult_segments = segments[segments['speaker_type'].isin(['FEM', 'MAL'])].sort_values(['segment_onset', 'segment_offset']).reset_index(drop = True)
    alice = am.get_segments(am.annotations[am.annotations['set'] == 'alice']).sort_values(['segment_onset', 'segment_offset']).reset_index(drop = True)
    
    pd.testing.assert_frame_equal(adult_segments[['phonemes', 'syllables', 'words']], alice[['phonemes', 'syllables', 'words']])

def test_clipping(project):
    am = AnnotationManager(project)

    input_annotations = pd.read_csv('examples/valid_raw_data/annotations/input.csv')
    am.import_annotations(input_annotations)
    am.read()

    start = 1981
    stop = 1984
    segments = am.get_segments(am.annotations[am.annotations['set'] == 'vtc_rttm'])
    segments = am.clip_segments(segments, start, stop)
    
    assert segments['segment_onset'].between(start, stop).all() and segments['segment_offset'].between(start, stop).all(), "segments not properly clipped"
    assert segments.shape[0] == 2, "got {} segments, expected 2".format(segments.shape[0])

def test_rename(project):
    am = AnnotationManager(project)

    input_annotations = pd.read_csv('examples/valid_raw_data/annotations/input.csv')
    am.import_annotations(input_annotations)
    am.read()
    its_count = am.annotations[am.annotations['set'] == 'new_its'].shape[0]

    am.rename_set('new_its', 'renamed')
    am.read()

    errors, warnings = am.validate()
    assert len(errors) == 0 and len(warnings) == 0, "malformed annotations detected"

    assert am.annotations[am.annotations['set'] == 'new_its'].shape[0] == 0
    assert am.annotations[am.annotations['set'] == 'renamed'].shape[0] == its_count


thresholds = [0, 0.5, 1]
@pytest.mark.parametrize('turntakingthresh', thresholds)
@pytest.mark.skipif(tuple(map(int, pd.__version__.split('.')[:2])) < (1,1), reason = "requires pandas>=1.1.0")
def test_vc_stats(project, turntakingthresh):
    am = AnnotationManager(project)
    am.import_annotations(pd.read_csv('examples/valid_raw_data/annotations/input.csv'))

    raw_rttm = 'example_metrics.rttm'
    segments = am.annotations[am.annotations['raw_filename'] == raw_rttm]
    
    vc = am.get_vc_stats(am.get_segments(segments), turntakingthresh = turntakingthresh).reset_index()
    truth_vc = pd.read_csv('tests/truth/vc_truth_{:.1f}.csv'.format(turntakingthresh))
   
    pd.testing.assert_frame_equal(
        vc.reset_index().sort_index(axis = 1).sort_values(vc.columns.tolist()),
        truth_vc.reset_index().sort_index(axis = 1).sort_values(vc.columns.tolist()),
        atol = 3
    )