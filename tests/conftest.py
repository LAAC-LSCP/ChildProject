"""Shared fixtures for the ChildProject test-suite.

Design rules followed by every test module:

* every test works on its own copy of an example dataset (``tmp_path``), never on
  ``examples/`` directly when it writes, and never on a shared ``output/`` folder,
  so tests are independent of each other and of execution order;
* the working directory is the repository root, so relative paths such as
  ``tests/data/...`` behave the same whatever directory pytest is started from;
* one test checks one behaviour; parametrization is used for variations of inputs.
"""
import logging
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"
VALID_DATASET = EXAMPLES / "valid_raw_data"
INVALID_DATASET = EXAMPLES / "invalid_raw_data"
TEST_DATA = REPO_ROOT / "tests" / "data"
TRUTH = REPO_ROOT / "tests" / "truth"

# importing the CLI configures the "ChildProject" logger (handlers + propagate=False);
# do it once here so the logging setup is identical whichever test module runs first
import ChildProject.cmdline as _cmdline  # noqa: E402,F401  (aliased: must not rebind the ChildProject class)


@pytest.fixture(scope="session", autouse=True)
def _repo_root_cwd():
    previous = os.getcwd()
    os.chdir(REPO_ROOT)
    yield
    os.chdir(previous)


@pytest.fixture(autouse=True)
def _propagate_package_logs():
    """ChildProject.cmdline sets propagate=False on the package logger, which would
    hide every ChildProject.* record from caplog. Re-enable propagation per test."""
    logger = logging.getLogger("ChildProject")
    previous = logger.propagate
    logger.propagate = True
    yield
    logger.propagate = previous


def copy_dataset(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst, symlinks=True)
    return dst


@pytest.fixture
def valid_path(tmp_path) -> Path:
    """fresh copy of examples/valid_raw_data"""
    return copy_dataset(VALID_DATASET, tmp_path / "valid_raw_data")


@pytest.fixture
def invalid_path(tmp_path) -> Path:
    """fresh copy of examples/invalid_raw_data"""
    return copy_dataset(INVALID_DATASET, tmp_path / "invalid_raw_data")


@pytest.fixture
def project(valid_path) -> ChildProject:
    """read ChildProject on a fresh copy of the valid dataset"""
    project = ChildProject(valid_path)
    project.read()
    return project


@pytest.fixture
def am(project) -> AnnotationManager:
    """AnnotationManager on the fresh valid dataset (index already read)"""
    return AnnotationManager(project)


@pytest.fixture
def long_am(project) -> AnnotationManager:
    """AnnotationManager whose recordings pretend to be long, so that annotations
    with large ranges can be imported without the 'offset beyond duration' check.
    The duration is overridden *after* building the manager: AnnotationManager re-reads
    the project (ChildProject.loaded is never set), which would discard the override."""
    am = AnnotationManager(project)
    project.recordings["duration"] = 100_000_000
    return am


@pytest.fixture
def lena_segments() -> pd.DataFrame:
    return pd.read_csv(TEST_DATA / "lena_its.csv")


@pytest.fixture
def aclew_segments() -> pd.DataFrame:
    return pd.read_csv(TEST_DATA / "aclew.csv")


@pytest.fixture
def conversation_segments() -> pd.DataFrame:
    """csv segments with 3 hand-labelled conversations (conv_count 1, 2, 3)"""
    segments = pd.read_csv(TEST_DATA / "csv.csv")
    segments.loc[2:4, "conv_count"] = 1
    segments.loc[8:9, "conv_count"] = 2
    segments.loc[10:11, "conv_count"] = 3
    return segments


def static_segments(data: pd.DataFrame, filename) -> pd.DataFrame:
    """import_function returning fixed segments whatever the raw file (use with functools.partial)"""
    return data.copy()


def import_static(am: AnnotationManager, segments: pd.DataFrame, set_name: str,
                  recording_filename: str = "sound.wav", range_offset: int = 30_000_000,
                  raw_filename: str = "file.its", fmt: str = "csv"):
    """import ``segments`` as set ``set_name`` covering [0, range_offset] of one recording"""
    from functools import partial

    return am.import_annotations(
        pd.DataFrame([{
            "set": set_name,
            "raw_filename": raw_filename,
            "time_seek": 0,
            "recording_filename": recording_filename,
            "range_onset": 0,
            "range_offset": range_offset,
            "format": fmt,
        }]),
        import_function=partial(static_segments, segments),
    )
