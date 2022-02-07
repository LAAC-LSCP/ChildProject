from ChildProject.projects import ChildProject


def test_valid_project():
    project = ChildProject("examples/valid_raw_data")
    errors, warnings = project.validate()

    assert len(errors) == 0, "valid input validation failed (expected to pass)"
    assert len(warnings) == 1, "expected 1 warning, got {}".format(len(warnings))


def test_invalid_project():
    project = ChildProject("examples/invalid_raw_data")
    errors, warnings = project.validate()

    expected_errors = [
        "examples/invalid_raw_data/metadata/children.csv: child_id '1' appears 2 times in lines [2,3], should appear once",
        "cannot find recording 'test_1_20200918.mp3' at 'examples/invalid_raw_data/recordings/raw/test_1_20200918.mp3'",
        "examples/invalid_raw_data/metadata/recordings.csv: 'USB' is not a permitted value for column 'recording_device_type' on line 2, should be any of [lena,usb,olympus,babylogger,unknown]",
    ]

    expected_warnings = [
        "examples/invalid_raw_data/metadata/recordings.csv: '2' does not pass callable test for column 'noisy_setting' on line 2",
        "file 'examples/invalid_raw_data/recordings/raw/test_1_2020091.mp3' not indexed.",
    ]
    assert sorted(expected_errors) == sorted(
        errors
    ), "errors do not match expected errors"
    assert sorted(expected_warnings) == sorted(
        warnings
    ), "warnings do not match expected warnings"

