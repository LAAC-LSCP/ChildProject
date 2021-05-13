import numpy as np
import pandas as pd

from ChildProject.metrics import gamma, segments_to_annotation, segments_to_grid, grid_to_vector, vectors_to_annotation_task, conf_matrix

def test_gamma():
    segments = pd.read_csv('tests/data/gamma.csv')

    value = gamma(
        segments,
        'speaker_type',
        alpha = 3,
        beta = 1,
        precision_level = 0.01
    )

    assert 0.28 <= value <= 0.31

def test_conf_matrix():
    segments = pd.read_csv('tests/data/confmatrix.csv')
    categories = ['CHI', 'FEM']

    matrix = conf_matrix(
        segments_to_grid(segments[segments['set'] == 'Alice'], 0, 20, 1, 'speaker_type', categories),
        segments_to_grid(segments[segments['set'] == 'Bob'], 0, 20, 1, 'speaker_type', categories),
        categories + ['overlap', 'none']
    )

    assert np.testing.assert_array_equal(
        matrix,
        [[5, 5, 0, 0], [0, 2, 0, 3], [0, 0, 0, 0], [0, 0, 0, 5]]
    )