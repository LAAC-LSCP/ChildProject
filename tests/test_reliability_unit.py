import pytest
import pandas as pd
import numpy as np
import os
from ChildProject.pipelines.reliability import ReliabilityAnalysis


# === 1. UNIT TESTS FOR `segments_to_multilabel_grid()` === #


def test_segments_to_multilabel_grid():
    """Test basic conversion of segments into a multilabel grid."""
    segments = pd.DataFrame(
        {
            "set": ["reference"] * 2,
            "segment_onset": [0, 1000],
            "segment_offset": [1000, 2000],
            "speaker_type": ["CHI", "FEM"],
        }
    )

    analyzer = ReliabilityAnalysis(None, "reference", "hypothesis", granularity=1000)
    grid = analyzer.segments_to_multilabel_grid(segments, "reference")

    assert len(grid) == 2
    assert "CHI" in grid[0]
    assert "FEM" in grid[1]


def test_segments_to_multilabel_grid_overlap():
    """Test overlapping speakers in the same time window."""
    segments = pd.DataFrame(
        {
            "set": ["reference"] * 3,
            "segment_onset": [0, 0, 1000],
            "segment_offset": [1000, 1000, 2000],
            "speaker_type": ["CHI", "FEM", "MAL"],
        }
    )

    analyzer = ReliabilityAnalysis(None, "reference", "hypothesis", granularity=1000)
    grid = analyzer.segments_to_multilabel_grid(segments, "reference")

    assert len(grid[0]) == 2
    assert "CHI" in grid[0] and "FEM" in grid[0]
    assert "MAL" in grid[1]


def test_segments_to_multilabel_grid_empty():
    """Test handling of empty segments."""
    segments = pd.DataFrame(
        columns=["set", "segment_onset", "segment_offset", "speaker_type"]
    )

    analyzer = ReliabilityAnalysis(None, "reference", "hypothesis", granularity=1000)
    grid = analyzer.segments_to_multilabel_grid(segments, "reference")

    assert len(grid) == 0
    assert isinstance(grid, np.ndarray)


def test_segments_to_multilabel_grid_invalid():
    """Test that invalid segment times raise an error."""
    segments = pd.DataFrame(
        {
            "set": ["reference"],
            "segment_onset": [1000],
            "segment_offset": [0],
            "speaker_type": ["CHI"],
        }
    )

    analyzer = ReliabilityAnalysis(None, "reference", "hypothesis")

    with pytest.raises(ValueError):
        analyzer.segments_to_multilabel_grid(segments, "reference")


def test_segments_to_multilabel_grid_unknown_speaker():
    """Test that unknown speakers are ignored."""
    segments = pd.DataFrame(
        {
            "set": ["reference"] * 3,
            "segment_onset": [0, 500, 1000],
            "segment_offset": [500, 1000, 1500],
            "speaker_type": ["CHI", "FEM", "UNKNOWN"],
        }
    )

    analyzer = ReliabilityAnalysis(
        None, "reference", "hypothesis", speakers=["CHI"], granularity=500
    )
    grid = analyzer.segments_to_multilabel_grid(segments, "reference")

    assert "CHI" in grid[0]
    assert "FEM" not in grid[1]
    assert "UNKNOWN" not in grid[2]


# === 2. UNIT TESTS FOR `generate_confusion_matrices()` === #


def test_confusion_matrix_generation(tmp_path):
    """Test that confusion matrices are generated correctly."""
    ref_grid = np.array([{"CHI"}, {"FEM"}, set()])
    hyp_grid = np.array([{"CHI"}, set(), {"FEM"}])

    analyzer = ReliabilityAnalysis(None, "reference", "hypothesis")
    analyzer.generate_confusion_matrices(ref_grid, hyp_grid, tmp_path)

    assert (tmp_path / "confusion_matrix_reference_vs_hypothesis.csv").exists()

    matrix = pd.read_csv(
        tmp_path / "confusion_matrix_reference_vs_hypothesis.csv", index_col=0
    )
    assert matrix.loc["CHI", "CHI"] == 1
    assert matrix.loc["FEM", "FEM"] == 0


# === 3. UNIT TESTS FOR `compute_classification_report()` === #


def test_classification_report(tmp_path):
    """Test that the classification report is generated correctly."""
    ref_grid = np.array([{"CHI"}, {"FEM"}, set()])
    hyp_grid = np.array([{"CHI"}, set(), {"FEM"}])

    analyzer = ReliabilityAnalysis(None, "reference", "hypothesis")
    analyzer.compute_classification_report(ref_grid, hyp_grid, tmp_path)

    report = pd.read_csv(tmp_path / "classification_report_reference_vs_hypothesis.csv")

    chi_row = report[report["Speaker"] == "CHI"].iloc[0]
    assert chi_row["Precision"] == 1.0
    assert chi_row["Recall"] == 1.0


def test_single_speaker():
    """Test reliability-analysis with only one speaker type."""
    segments = pd.DataFrame(
        {
            "set": ["reference"] * 5,
            "segment_onset": [0, 100, 200, 300, 400],
            "segment_offset": [100, 200, 300, 400, 500],
            "speaker_type": ["CHI"] * 5,  # only one speaker
        }
    )

    analyzer = ReliabilityAnalysis(None, "reference", "hypothesis", granularity=100)
    grid = analyzer.segments_to_multilabel_grid(segments, "reference")

    assert all(grid[i] == {"CHI"} for i in range(len(grid)))
