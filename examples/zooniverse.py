#!/usr/bin/env python3
from ChildProject.projects import ChildProject
from ChildProject.pipelines.zooniverse import ZooniversePipeline
from ChildProject.pipelines.samplers import HighVolubilitySampler

import argparse
import os

parser = argparse.ArgumentParser(description='import and convert VTC annotations into the project')
parser.add_argument("--source", help = "project path", required = True)
parser.add_argument("--chunks-destination", help = "chunks destination", required = True)
parser.add_argument("--set", help = "annotation set", default = 'its')
parser.add_argument("--project-id", help = "id of the zooniverse project to upload the chunks to", default = '')
args = parser.parse_args()

project = ChildProject(args.source)
project.read()

sampler = HighVolubilitySampler(
    project,
    annotation_set = args.set,
    metric = 'cvc',
    windows_length = 60*1000,
    windows_count = 10
)
sampler.sample()
sampler.segments.to_csv('segments.csv')

zooniverse = ZooniversePipeline()
chunks_path = zooniverse.extract_chunks(
    path = project.path,
    destination = args.chunks_destination,
    keyword = 'example',
    segments = 'segments.csv',
    chunks_length = 500,
    chunks_min_amount = 2,
    threads = 4
)

zooniverse.upload_chunks(
    chunks = chunks_path,
    project_id = args.project_id,
    set_prefix = 'example',
    batches = 2
)