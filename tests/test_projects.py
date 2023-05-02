from ChildProject.projects import ChildProject
import pandas as pd
import pytest

def test_enforce_dtypes():
    project = ChildProject("examples/valid_raw_data", enforce_dtypes=True)
    project.read()

    assert project.recordings["child_id"].dtype.kind == "O"
    assert project.children["child_id"].dtype.kind == "O"

    project = ChildProject("examples/valid_raw_data", enforce_dtypes=False)
    project.read()

    assert project.recordings["child_id"].dtype.kind == "i"
    assert project.children["child_id"].dtype.kind == "i"
    
@pytest.mark.parametrize("idis,rshape,cshape", 
                         [(True,2,1),
                         (False,3,2),
                         ])
def test_ignore_discarded(idis,rshape,cshape):
    project = ChildProject("examples/valid_raw_data", ignore_discarded=idis)
    project.read()
    
    assert project.recordings.shape[0] == rshape
    assert project.children.shape[0] == cshape


def test_compute_ages():
    project = ChildProject("examples/valid_raw_data")
    project.read()

    project.recordings["age"] = project.compute_ages()
    project.recordings["age_days"] = project.compute_ages(age_format='days')
    project.recordings["age_weeks"] = project.compute_ages(age_format='weeks')
    project.recordings["age_years"] = project.compute_ages(age_format='years')

    truth = pd.read_csv("tests/truth/ages.csv").set_index("line")

    pd.testing.assert_frame_equal(
        project.recordings[["child_id", "age","age_days","age_weeks","age_years"]], truth[["child_id", "age","age_days","age_weeks","age_years"]]
    )

