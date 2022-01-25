from ChildProject.projects import ChildProject
import pandas as pd


def test_enforce_dtypes():
    project = ChildProject("examples/valid_raw_data", enforce_dtypes=True)
    project.read()

    assert project.recordings["child_id"].dtype.kind == "O"
    assert project.children["child_id"].dtype.kind == "O"

    project = ChildProject("examples/valid_raw_data", enforce_dtypes=False)
    project.read()

    assert project.recordings["child_id"].dtype.kind == "i"
    assert project.children["child_id"].dtype.kind == "i"


def test_compute_ages():
    project = ChildProject("examples/valid_raw_data")
    project.read()

    project.recordings["age"] = project.compute_ages()

    truth = pd.read_csv("tests/truth/ages.csv").set_index("line")

    pd.testing.assert_frame_equal(
        project.recordings[["child_id", "age"]], truth[["child_id", "age"]]
    )

