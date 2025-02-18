import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from ChildProject.pipelines.reliability import ReliabilityAnalysis


def test_reliability_pipeline(tmp_path):
    """Test the full pipeline execution with a simple dataset."""
    segments = pd.DataFrame(
        {
            "set": ["reference"] * 3,
            "segment_onset": [0, 500, 1000],
            "segment_offset": [500, 1000, 1500],
            "speaker_type": ["CHI", "FEM", "MAL"],
        }
    )

    hypothesis_segments = pd.DataFrame(
        {
            "set": ["hypothesis"] * 3,
            "segment_onset": [0, 500, 1000],
            "segment_offset": [500, 1000, 1500],
            "speaker_type": ["CHI", "OCH", "MAL"],
        }
    )

    output_dir = tmp_path / "reliability_results"
    os.makedirs(output_dir, exist_ok=True)

    analyzer = ReliabilityAnalysis(None, "reference", "hypothesis", granularity=500)

    ref_grid = analyzer.segments_to_multilabel_grid(segments, "reference")
    hyp_grid = analyzer.segments_to_multilabel_grid(hypothesis_segments, "hypothesis")

    analyzer.generate_confusion_matrices(ref_grid, hyp_grid, output_dir)
    analyzer.compute_classification_report(ref_grid, hyp_grid, output_dir)

    assert (output_dir / "confusion_matrix_reference_vs_hypothesis.csv").exists()
    assert (output_dir / "classification_report_reference_vs_hypothesis.csv").exists()


def test_reliability_pipeline_large_dataset(tmp_path):
    """Test the pipeline with a large dataset to check performance."""
    num_segments = 10000
    segments = pd.DataFrame(
        {
            "set": ["reference"] * num_segments,
            "segment_onset": np.arange(0, num_segments * 100, 100),
            "segment_offset": np.arange(100, (num_segments + 1) * 100, 100),
            "speaker_type": np.random.choice(
                ["CHI", "FEM", "MAL", "OCH"], num_segments
            ),
        }
    )

    hypothesis_segments = segments.copy()
    hypothesis_segments["set"] = "hypothesis"

    output_dir = tmp_path / "reliability_large"
    os.makedirs(output_dir, exist_ok=True)

    analyzer = ReliabilityAnalysis(None, "reference", "hypothesis", granularity=100)

    ref_grid = analyzer.segments_to_multilabel_grid(segments, "reference")
    hyp_grid = analyzer.segments_to_multilabel_grid(hypothesis_segments, "hypothesis")

    analyzer.generate_confusion_matrices(ref_grid, hyp_grid, output_dir)
    analyzer.compute_classification_report(ref_grid, hyp_grid, output_dir)

    assert (output_dir / "confusion_matrix_reference_vs_hypothesis.csv").exists()
    assert (output_dir / "classification_report_reference_vs_hypothesis.csv").exists()
