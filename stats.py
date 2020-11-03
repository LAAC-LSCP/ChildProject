#!/usr/bin/env python3
from ChildProject.projects import ChildProject
import argparse
import sys

parser = argparse.ArgumentParser(description='Get project stats')
parser.add_argument("--source", help = "source data path", required = True)
parser.add_argument("--stats", help = "stats to retrieve (comma-separated)", required = False, default = "")
args = parser.parse_args()

project = ChildProject(args.source)

errors, warnings = project.validate_input_data()

if len(errors) > 0:
    print("validation failed, {} error(s) occured".format(len(errors)), file = sys.stderr)
    sys.exit(1)

stats = project.get_stats()
args.stats = args.stats.split(',') if args.stats else []

for stat in stats:
    if not args.stats or stat in args.stats:
        print("{}: {}".format(stat, stats[stat]))