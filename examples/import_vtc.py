#!/usr/bin/env python3
from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager

import argparse
import os

parser = argparse.ArgumentParser(description='import and convert VTC annotations into the project')
parser.add_argument("--source", help = "project path", required = True)
args = parser.parse_args()

project = ChildProject(args.source)
am = AnnotationManager(project)

input = project.recordings[['filename']]
input.rename(columns = {'filename': 'recording_filename'}, inplace = True)
input = input[input['recording_filename'] != 'NA']
input['set'] = 'vtc'
input['time_seek'] = 0
input['range_onset'] = 0
input['range_offset'] = 0
input['raw_filename'] = input['recording_filename'].apply(lambda s: os.path.join('vtc', s + '.rttm'))
input['format'] = 'vtc_rttm'

am.import_annotations(input)