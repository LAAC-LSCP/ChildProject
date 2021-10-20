#!/usr/bin/env python3
from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager

import argparse
import os

parser = argparse.ArgumentParser(description='import and convert VTC annotations into the project')
parser.add_argument("--source", help = "project path", required = True)
parser.add_argument("--set", help = "annotation set. the rttm files should lie in <source>/annotations/<set>/raw/", default = 'vtc')
parser.add_argument("--overwrite", help = "project path", dest = 'overwrite', action = 'store_true')
args = parser.parse_args()

project = ChildProject(args.source)
am = AnnotationManager(project)

if args.overwrite:
    am.remove_set(args.set)

input = project.recordings[['recording_filename', 'duration']]
input = input[input['recording_filename'] != 'NA']
input['set'] = args.set
input['time_seek'] = 0
input['range_onset'] = 0
input['range_offset'] = input['duration']
input['raw_filename'] = input['recording_filename'].apply(lambda s: os.path.splitext(s)[0] + '.rttm')
input['format'] = 'vtc_rttm'

am.import_annotations(input, threads = 4)