from ChildProject.projects import ChildProject, RecordingProfile
from ChildProject.annotations import AnnotationManager
from ChildProject.tables import IndexTable
import pandas as pd
import numpy as np
import os

def test_import():
    project = ChildProject("examples/valid_raw_data")
    project.import_data("output/annotations")
    project = ChildProject("output/annotations")
    am = AnnotationManager(project)

    input_annotations = pd.read_csv('examples/valid_raw_data/raw_annotations/input.csv')

    am.import_annotations(input_annotations)
    am.read()
    
    assert am.annotations.shape[0] == input_annotations.shape[0], "imported annotations length does not match input"

    assert all([
        os.path.exists(os.path.join(project.path, 'annotations', f))
        for f in am.annotations['annotation_filename'].tolist()
    ]), "some annotations are missing"

    errors, warnings = am.validate()
    assert len(errors) == 0 and len(warnings) == 0, "malformed annotations detected"

    for dataset in ['eaf', 'textgrid']:
        annotations = am.annotations[am.annotations['set'] == dataset]
        segments = am.get_segments(annotations)
        segments.drop(columns = annotations.columns, inplace = True)

        pd.testing.assert_frame_equal(
            segments.sort_index(axis = 1).sort_values(segments.columns.tolist()).reset_index(drop = True),
            pd.read_csv('tests/truth/{}.csv'.format(dataset)).sort_index(axis = 1).sort_values(segments.columns.tolist()).reset_index(drop = True),
            check_less_precise = True
        )

def test_intersect():
    project = ChildProject("examples/valid_raw_data")
    project.import_data("output/metrics")
    project = ChildProject("output/metrics")
    am = AnnotationManager(project)

    input_annotations = pd.read_csv('examples/valid_raw_data/raw_annotations/intersect.csv')

    am.import_annotations(input_annotations)
    am.read()

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