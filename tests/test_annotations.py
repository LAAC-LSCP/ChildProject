from ChildProject.projects import ChildProject, RecordingProfile
from ChildProject.VTCAnnotation import VTCAnnotation
import pandas as pd
import os

def test_vtc():
    project = ChildProject("examples/valid_raw_data")
    project.import_data("output/vtc")
    project.convert_recordings(RecordingProfile(name = '16kHz', sampling = 16000))

    inputs = [
        pd.read_csv('examples/vtc_input1.csv'),
        pd.read_csv('examples/vtc_input2.csv')
    ]

    vtc = VTCAnnotation(project)

    for input in inputs:
        vtc.pre_process('vtc_test', input, '16kHz')
        vtc.process(options = ['--device=gpu'])
        vtc.post_process()

    index = pd.read_csv("output/vtc/annotations/vtc_test/index.csv")
    
    assert sum(input.shape[0] for input in inputs) == index.shape[0], "input index is not complete"
    assert all(index['completed'].tolist()), "some task(s) failed to complete"
    assert all([
        os.path.exists(os.path.join("output/vtc/annotations/vtc_test/output", f))
        for f in index['output_filename'].tolist()
    ]), "output files are missing"

