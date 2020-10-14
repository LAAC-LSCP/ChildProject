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
    assert len(errors) > 0 or len(warnings) > 0, "malformed annotations detected"

    annotations = am.annotations[am.annotations['set'] == 'textgrid']
    segments = annotations['annotation_filename'].map(lambda f: pd.read_csv(os.path.join(project.path, 'annotations', f))).tolist()
    segments = pd.concat(segments)

    pd.testing.assert_frame_equal(segments.sort_index(axis=1), pd.read_csv('tests/truth/textgrid.csv').sort_index(axis=1))

