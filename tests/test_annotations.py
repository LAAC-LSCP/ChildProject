from ChildProject.projects import ChildProject, RecordingProfile
from ChildProject.annotations import AnnotationManager
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
    
    assert am.annotations.shape[0] == input_annotations.shape[0], "imported annotations length does not match input"

    assert all([
        os.path.exists(os.path.join(project.path, 'annotations', f))
        for f in am.annotations['annotation_filename'].tolist()
    ]), "some annotations are missing"

    segments = am.annotations['annotation_filename'].map(lambda f: pd.read_csv(os.path.join(project.path, 'annotations', f))).tolist()
    segments = pd.concat(segments)
    CHI_segments = segments[(segments['speaker_type'] == 'CHI') & (segments['ling_type'] == 1)]

    raise Exception((CHI_segments['segment_offset']-CHI_segments['segment_onset']).sum())
    
    assert np.isclose((CHI_segments['segment_offset']-CHI_segments['segment_onset']).sum(), 2.867) == True



