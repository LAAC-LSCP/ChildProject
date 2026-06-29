from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
import os

def test_valid_project():
    path = os.path.normpath("examples/valid_raw_data")
    project = ChildProject(path)
    errors, warnings = project.validate()

    assert len(errors) == 0, "valid input validation failed (expected to pass)"
    assert len(warnings) == 1, "expected 3 warnings, got {}".format(len(warnings))


def test_invalid_project():
    project = ChildProject("examples/invalid_raw_data")
    errors, warnings = project.validate()
    
    am = AnnotationManager(project)
    
    errors.extend(am.errors)
    warnings.extend(am.warnings)

    expected_errors = [
        os.path.normpath("examples/invalid_raw_data/metadata/children.csv")+ ": Duplicated values when it should be unique for column child_id, values {'1'} on lines {2, 3} appear multiple times",
        os.path.normpath("examples/invalid_raw_data/metadata/recordings.csv")+ ": \n2 validation errors for RecordingModel\nrecording_device_type\n  String should match pattern 'lena|usb|olympus|babylogger|izyrec|unknown' [type=string_pattern_mismatch, input_value='USB', input_type=str]\n    For further information visit https://errors.pydantic.dev/2.13/v/string_pattern_mismatch\nnoisy_setting\n  Input should be a valid boolean, unable to interpret input [type=bool_parsing, input_value=2, input_type=int]\n    For further information visit https://errors.pydantic.dev/2.13/v/bool_parsing",
        "'recording_filename' values ['test_1_20200918.mp3', 'test_1_is_not_here.wav'] in recordings table on lines [2, 5] cannot be found in the filesystem.",
        'Age at recording is negative in recordings on line 3 (-15.4 months). Check date_iso for that recording and child_dob for the corresponding child.', 
        'Age at recording is negative in recordings on line 4 (-15.4 months). Check date_iso for that recording and child_dob for the corresponding child.', 
        'Age at recording is negative in recordings on line 5 (-15.4 months). Check date_iso for that recording and child_dob for the corresponding child.',
        "duplicate reference to annotations/vtc_rttm/converted/sound_1980000_1990000.csv (appears 2 times)",
        "annotation index does not verify range_offset > range_onset >= 0 for set <ranges>, annotation filename <sound_0_300000.csv>",
        "annotation index has an offset higher than recorded duration of the audio <textgrid>, annotation filename <sound_0_40000000.csv>",

                       ]

    expected_warnings = [
        "Metadata files for sets ['alice', 'ranges', 'textgrid', 'vtc_rttm'] could not be found, they should be created as annotations/<set>/metannots.yml",
        "files {'test_1_2020091.mp3'} not indexed in recording_filename column",
    ]
    assert sorted(expected_errors) == sorted(
        errors
    ), "errors do not match expected errors"
    assert sorted(expected_warnings) == sorted(
        warnings
    ), "warnings do not match expected warnings"

