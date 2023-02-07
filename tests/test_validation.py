from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
import os

def test_valid_project():
    path = os.path.normpath("examples/valid_raw_data")
    project = ChildProject(path)
    errors, warnings = project.validate()

    assert len(errors) == 0, "valid input validation failed (expected to pass)"
    assert len(warnings) == 3, "expected 1 warning, got {}".format(len(warnings))


def test_invalid_project():
    project = ChildProject("examples/invalid_raw_data")
    errors, warnings = project.validate()
    
    am = AnnotationManager(project)
    
    errors.extend(am.errors)
    warnings.extend(am.warnings)

    expected_errors = [
        os.path.normpath("examples/invalid_raw_data/metadata/children.csv")+ ": child_id '1' appears 2 times in lines [2,3], should appear once",
        "cannot find recording 'test_1_20200918.mp3' at "+os.path.normpath("'examples/invalid_raw_data/recordings/raw/test_1_20200918.mp3'"),
        os.path.normpath("examples/invalid_raw_data/metadata/recordings.csv")+ ": 'USB' is not a permitted value for column 'recording_device_type' on line 2, should be any of [lena,usb,olympus,babylogger,unknown]",
        "Age at recording is negative in recordings on line 3 (-15.4 months). Check date_iso for that recording and child_dob for the corresponding child.",
        "duplicate reference to annotations/vtc_rttm/converted/sound_1980000_1990000.csv (appears 2 times)",
        "overlaps in the annotation index for the following [set, annotation_filename] list: [['textgrid', 'sound_0_10000.csv'], ['textgrid', 'sound_0_300000.csv'], ['textgrid', 'sound_0_40000000.csv'], ['vtc_rttm', 'sound_1980000_1990000.csv'], ['vtc_rttm', 'sound_1980000_1990000.csv']]",
        "annotation index does not verify range_offset > range_onset >= 0 for set <ranges>, annotation filename <sound_0_300000.csv>",
    ]

    expected_warnings = [
        os.path.normpath("examples/invalid_raw_data/metadata/recordings.csv")+ ": '2' does not pass callable test for column 'noisy_setting' on line 2",
        "file '"+ os.path.normpath("examples/invalid_raw_data/recordings/raw/test_1_2020091.mp3")+"' not indexed.",
    ]
    assert sorted(expected_errors) == sorted(
        errors
    ), "errors do not match expected errors"
    assert sorted(expected_warnings) == sorted(
        warnings
    ), "warnings do not match expected warnings"

