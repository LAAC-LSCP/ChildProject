from ChildProject.ChildProject import ChildProject

def test_valid_project():
    project = ChildProject("examples/valid_raw_data")
    errors, warnings = project.validate_input_data()

    assert len(errors) == 0, "valid input validation failed (expected to pass)"
    assert len(warnings) == 1, "expected 1 warning, got {}".format(len(warnings))

def test_invalid_project():
    project = ChildProject("examples/invalid_raw_data")
    errors, warnings = project.validate_input_data()
    assert len(errors) > 0, "valid input validation passed (expected to fail)"