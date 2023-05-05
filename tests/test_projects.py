from ChildProject.projects import ChildProject
import pandas as pd
import pytest
import shutil
import os

TEST_DIR = os.path.join("output", "projects")

@pytest.fixture(scope="function")
def project(request):
    if os.path.exists(TEST_DIR):
#        shutil.copytree(src="examples/valid_raw_data", dst="output/annotations")
        shutil.rmtree(TEST_DIR)
    shutil.copytree(src="examples/valid_raw_data", dst=TEST_DIR)
    
    project = ChildProject(TEST_DIR)
    yield project

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

@pytest.mark.parametrize("error,chi_lines,rec_lines", 
                         [(None,[],[]),
                         (ValueError,['test2,3,2018-02-02,0'],[]),
                         ])
def test_projects_read(project, error, chi_lines, rec_lines):
    
    if chi_lines:
        chi_path = os.path.join(TEST_DIR, "metadata","children.csv")
        with open(chi_path, "a") as f:
            for l in chi_lines:
                f.write(str(l) + '\n')
    if rec_lines:
        rec_path = os.path.join(TEST_DIR, "metadata","recordings.csv")
        with open(rec_path, "a") as f:
            for l in chi_lines:
                f.write(str(l) + '\n')
    if error:
        with pytest.raises(error):
            project.read()
    else:
        project.read()