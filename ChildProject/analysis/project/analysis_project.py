from typing import Any
import datalad.api as dl
from pathlib import Path
from .constants import SIBLING_NAME_GITHUB, SIBLING_NAME_GIN


class AnalysisProject:
    """
    A class representing an analysis project.

    This class is responsible for managing the creation and management of analysis projects.

    Let's see we want a project to be able to be made
    When made folder has the project name
    Also needs to call Datalad API
    Also needs a way to add remotes GIN and GitHub
    Needs some sort of validation mechanism?
    And some sort of installation
    """

    _name: str
    _project_path: Path
    _dataset: dl.Dataset

    def __init__(self, project_name: str, base_dir: Path = None) -> None:
        """
        Initialize an AnalysisProject instance.

        :param project_name: The name of the project.
        :param base_dir: The base directory where the project folder will be created. Defaults to the present working directory
        """
        if base_dir is None:
            base_dir = Path.cwd()

        self._name = project_name
        self._project_path = base_dir / self._name

        if self._project_exists():
            self._dataset = dl.Dataset(self._project_path)

    def create_project(self) -> None:
        if self._project_exists():
            raise IsADirectoryError(f"Project '{self._name}' already exists")

        self._create_dataset()

    @property
    def project_dir(self) -> Path:
        """
        Get the project directory path.

        :return: The path to the project directory.
        """
        return self._project_path
    
    @property
    def id(self) -> str:
        """
        Datalad id of the underlying dataset
        """
        return self._dataset.id

    def _create_dataset(self) -> None:
        # Run built-in yoda procedure. Use `datalad run-procedure --discover` to see the script
        dl.create(path=self._project_path, cfg_proc="yoda")

        self._dataset = dl.Dataset(path=(self._project_path))

        self._make_subdir("outputs")

    def _make_subdir(self, subdir_name: str, annex=True) -> None:
        """
        Add a subdirectory to the main analysis project

        :param subdir_name: Name of the subdirectory to create.
        :param annex: Whether to create the subdirectory as an annexed directory. Defaults to `True`
        """

        subdir: Path = self._project_path / subdir_name
        subdir.mkdir(exist_ok=False)

        if not annex:
            return

        dl.no_annex(dataset=self._dataset, ref_dir=".", pattern=[f"{subdir_name}/**"])

    def add_github_sibling(
        self,
        repo_name: str,
    ) -> None:
        self._dataset.create_sibling_github(
            reponame=repo_name, name=SIBLING_NAME_GITHUB
        )

    def add_gin_sibling(
        self,
        repo_name: str,
    ) -> None:
        self._dataset.create_sibling_gin(reponame=repo_name, name=SIBLING_NAME_GIN)

    def _project_exists(self) -> bool:
        return self._project_path.exists()

    def validate_project(self) -> Any:
        # Would return a type of error object or bool at some point
        raise NotImplementedError()
