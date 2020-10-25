from ChildProject.projects import ChildProject

def test_valid_project():
    project = ChildProject("examples/valid_raw_data")
    errors, warnings = project.validate_input_data()

    assert len(errors) == 0, "valid input validation failed (expected to pass)"
    assert len(warnings) == 1, "expected 1 warning, got {}".format(len(warnings))

def test_invalid_project():
    project = ChildProject("examples/invalid_raw_data")
    errors, warnings = project.validate_input_data()
    
    expected_errors = [
        "child_id '1' appears 2 times in lines [2,3], should appear once",
        "cannot find recording 'test_1_20200918.mp3'"
    ]
    
    assert sorted(expected_errors) == sorted(errors), "errors do not match expected errors"

    