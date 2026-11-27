from ChildProject.analysis import AnalysisProject
from pathlib import Path
import pytest


@pytest.fixture(scope="module")
def analysis_project(tmp_path_factory) -> AnalysisProject:
    tmp_path = tmp_path_factory.mktemp("analysis_project")
    project = AnalysisProject("my_project", tmp_path)
    project.create_project()

    return project


def test_project(analysis_project):
    project_dir: Path = analysis_project.project_dir

    assert project_dir.exists()
    assert (project_dir / "outputs").exists()


def test_project_datalad_folder(analysis_project):
    datalad_dir: Path = analysis_project.project_dir / ".datalad"
    gitattributes: Path = datalad_dir / ".gitattributes"
    config: Path = datalad_dir / "config"

    assert gitattributes.read_text().splitlines() == ["config annex.largefiles=nothing"]
    assert config.read_text().splitlines() == [
        '[datalad "dataset"]',
        f"\tid = {analysis_project.id}",  # Normally you would seed before testing but Datalad uses OS-level random seed generation
    ]
