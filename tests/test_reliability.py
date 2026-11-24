"""reliability / agreement helpers (ChildProject.metrics)"""
import numpy as np
import pandas as pd
import pytest
from pyannote.core import Annotation, Segment

from ChildProject.metrics import (
    conf_matrix,
    gamma,
    grid_to_vector,
    pyannote_metric,
    segments_to_annotation,
    segments_to_grid,
    vectors_to_annotation_task,
)
from ChildProject.tables import MissingColumnsException

from conftest import TEST_DATA

GRID_TRUTH = np.array([
    [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1],
    [0, 0, 0, 1], [1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1],
])  # columns: CHI, FEM, overlap, none


@pytest.fixture
def grid_segments():
    return pd.read_csv(TEST_DATA / "grid.csv")


# ============================================================ segments_to_annotation
def test_segments_to_annotation():
    segments = pd.DataFrame({"segment_onset": [0, 10], "segment_offset": [5, 20], "speaker_type": ["CHI", "FEM"]})
    annotation = segments_to_annotation(segments, "speaker_type")
    assert isinstance(annotation, Annotation)
    assert sorted((s.start, s.end, label) for s, _, label in annotation.itertracks(yield_label=True)) == [
        (0, 5, "CHI"), (10, 20, "FEM")]


def test_segments_to_annotation_requires_columns():
    with pytest.raises(MissingColumnsException):
        segments_to_annotation(pd.DataFrame({"segment_onset": [0]}), "speaker_type")


# ============================================================ pyannote_metric
def test_pyannote_metric_compares_reference_and_hypothesis_sets():
    segments = pd.DataFrame({"set": ["ref", "hyp"], "segment_onset": [0, 0], "segment_offset": [10, 5],
                             "speaker_type": ["CHI", "CHI"]})
    calls = []

    def metric(reference, hypothesis, detailed):
        calls.append(detailed)
        return reference.get_timeline().duration(), hypothesis.get_timeline().duration()

    assert pyannote_metric(segments, "ref", "hyp", metric, "speaker_type") == (10, 5)
    assert calls == [True]


def test_pyannote_metric_requires_set_column():
    with pytest.raises(MissingColumnsException):
        pyannote_metric(pd.DataFrame({"segment_onset": [0]}), "a", "b", None, "speaker_type")


# ============================================================ segments_to_grid
@pytest.mark.parametrize("overlap,none,columns", [
    (True, True, [0, 1, 2, 3]),
    (False, True, [0, 1, 3]),
    (True, False, [0, 1, 2]),
    (False, False, [0, 1]),
])
def test_segments_to_grid(grid_segments, overlap, none, columns):
    grid = segments_to_grid(grid_segments, 0, 10, 1, "speaker_type", ["CHI", "FEM"], overlap=overlap, none=none)
    np.testing.assert_array_equal(grid, GRID_TRUTH[:, columns])


def test_segments_to_grid_default_has_none_but_no_overlap(grid_segments):
    grid = segments_to_grid(grid_segments, 0, 10, 1, "speaker_type", ["CHI", "FEM"])
    np.testing.assert_array_equal(grid, GRID_TRUTH[:, [0, 1, 3]])


def test_segments_to_grid_ignores_unknown_categories(grid_segments):
    grid = segments_to_grid(grid_segments, 0, 10, 1, "speaker_type", ["CHI"], none=False)
    np.testing.assert_array_equal(grid, GRID_TRUTH[:, [0]])


def test_segments_to_grid_timescale_and_offset():
    segments = pd.DataFrame({"segment_onset": [1000], "segment_offset": [1300], "speaker_type": ["CHI"]})
    grid = segments_to_grid(segments, 1000, 1500, 100, "speaker_type", ["CHI"], none=False)
    assert grid[:, 0].tolist() == [1, 1, 1, 0, 0]


def test_segments_to_grid_partial_last_unit():
    segments = pd.DataFrame({"segment_onset": [0], "segment_offset": [10], "speaker_type": ["CHI"]})
    assert segments_to_grid(segments, 0, 25, 10, "speaker_type", ["CHI"]).shape == (3, 2)


def test_segments_to_grid_categories_are_compared_as_strings():
    segments = pd.DataFrame({"segment_onset": [0], "segment_offset": [2], "vcm_type": [1]})
    grid = segments_to_grid(segments, 0, 2, 1, "vcm_type", [1], none=False)
    assert grid[:, 0].tolist() == [1, 1]


def test_segments_to_grid_does_not_modify_input(grid_segments):
    before = grid_segments.copy()
    segments_to_grid(grid_segments, 0, 10, 1, "speaker_type", ["CHI", "FEM"])
    pd.testing.assert_frame_equal(grid_segments, before)


def test_segments_to_grid_requires_columns():
    with pytest.raises(MissingColumnsException):
        segments_to_grid(pd.DataFrame({"segment_onset": [0]}), 0, 1, 1, "speaker_type", ["CHI"])


# ============================================================ grid_to_vector
def test_grid_to_vector(grid_segments):
    grid = segments_to_grid(grid_segments, 0, 10, 1, "speaker_type", ["CHI", "FEM"], overlap=True, none=True)
    vector = grid_to_vector(grid, ["CHI", "FEM", "overlap", "none"])
    assert vector.tolist() == ["CHI", "CHI", "FEM", "FEM", "none", "none", "overlap", "overlap", "FEM", "none"]


def test_grid_to_vector_last_active_class_wins():
    assert grid_to_vector(np.array([[1, 1, 0], [1, 0, 1]]), ["a", "b", "c"]).tolist() == ["b", "c"]


# ============================================================ conf_matrix
def test_conf_matrix():
    segments = pd.read_csv(TEST_DATA / "confmatrix.csv")
    grid = lambda s: segments_to_grid(segments[segments["set"] == s], 0, 20, 1, "speaker_type", ["CHI", "FEM"],
                                      overlap=True, none=True)
    np.testing.assert_array_equal(conf_matrix(grid("Bob"), grid("Alice")),
                                  np.array([[5, 5, 0, 0], [0, 2, 0, 3], [0, 0, 0, 0], [0, 0, 0, 5]]))


def test_conf_matrix_identical_grids_is_diagonal_for_exclusive_classes():
    grid = np.array([[1, 0], [0, 1], [0, 1]])
    np.testing.assert_array_equal(conf_matrix(grid, grid), np.array([[1, 0], [0, 2]]))


# ============================================================ vectors_to_annotation_task
def test_vectors_to_annotation_task_alpha():
    segments = pd.read_csv(TEST_DATA / "alpha.csv")
    categories = list(segments["speaker_type"].unique())
    vectors = [
        grid_to_vector(segments_to_grid(segments[segments["set"] == s], 0, segments["segment_offset"].max(), 1,
                                        "speaker_type", categories, overlap=True, none=True), categories + ["overlap", "none"])
        for s in segments["set"].unique()
    ]
    assert vectors_to_annotation_task(*vectors, drop=["none"]).alpha() == pytest.approx(0.743421052632, rel=0.001, abs=0.0001)


def test_vectors_to_annotation_task_data():
    task = vectors_to_annotation_task(np.array(["a", "b"]), np.array(["a", "a"]))
    assert sorted(task.data, key=lambda d: (d["coder"], d["item"])) == [
        {"coder": 0, "labels": "a", "item": 0}, {"coder": 0, "labels": "b", "item": 1},
        {"coder": 1, "labels": "a", "item": 0}, {"coder": 1, "labels": "a", "item": 1},
    ]


def test_vectors_to_annotation_task_drop():
    task = vectors_to_annotation_task(np.array(["a", "none"]), np.array(["a", "a"]), drop=["none"])
    assert len(task.data) == 3


# ============================================================ gamma (optional dependency)
def test_gamma():
    pytest.importorskip("pygamma_agreement", reason="pygamma-agreement support is optional")
    assert 0.39 <= gamma(pd.read_csv(TEST_DATA / "gamma.csv"), "speaker_type", alpha=3, beta=1, precision_level=0.01) <= 0.44


def test_gamma_requires_columns():
    with pytest.raises(MissingColumnsException):
        gamma(pd.DataFrame({"segment_onset": [0]}), "speaker_type")
