#!/usr/bin/env python3

from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from ChildProject.metrics import gamma, segments_to_grid, grid_to_vector, vectors_to_annotation_task

import argparse

parser = argparse.ArgumentParser(description = 'compute agreement measures for all given annotators over a whole dataset')
parser.add_argument('path', help = 'path to the dataset')
parser.add_argument('--sets', nargs = '+', help = 'sets to include')
args = parser.parse_args()

speakers = ['CHI', 'OCH', 'FEM', 'MAL']

project = ChildProject(args.path)
am = AnnotationManager(project)
am.read()

intersection = AnnotationManager.intersection(am.annotations, args.sets)
segments = am.get_collapsed_segments(intersection)

segments = segments[segments['speaker_type'].isin(speakers)]

vectors = [
    grid_to_vector(
        segments_to_grid(
            segments[segments['set'] == s],
            0,
            segments['segment_offset'].max(),
            100,
            'speaker_type',
            speakers,
            none = False
        ),
        speakers
    )
    for s in args.sets
]

task = vectors_to_annotation_task(*vectors)


alpha = task.alpha()
print(f'Krippendorff\'s alpha = {alpha:.2f}')

kappa = task.multi_kappa()
print(f'Fleiss\' kappa = {kappa:.2f}')

gamma = gamma(segments, 'speaker_type')
print(f'Mathet et al.\'s gamma = {gamma:.2f}')