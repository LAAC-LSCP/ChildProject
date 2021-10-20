from ChildProject.projects import ChildProject

def test_enforce_dtypes():
    project = ChildProject("examples/valid_raw_data", enforce_dtypes=True)
    project.read()

    assert project.recordings['child_id'].dtype.kind == 'O'
    assert project.children['child_id'].dtype.kind == 'O'

    project = ChildProject("examples/valid_raw_data", enforce_dtypes=False)
    project.read()

    assert project.recordings['child_id'].dtype.kind == 'i'
    assert project.children['child_id'].dtype.kind == 'i'
