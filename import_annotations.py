#!/usr/bin/env python3
from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
import argparse
import pandas as pd
import sys

parser = argparse.ArgumentParser(description='import and convert annotations into the project')
parser.add_argument("--source", help = "project path", required = True)
parser.add_argument("--annotations", help = "path to input annotations index (csv)", default = "")

for col in AnnotationManager.INDEX_COLUMNS:
    if col.generated:
        continue

    parser.add_argument("--{}".format(col.name), help = col.description, type = str, default = None)

args = parser.parse_args()

project = ChildProject(args.source)
errors, warnings = project.validate_input_data()

if len(errors) > 0:
    print("validation failed, {} error(s) occured".format(len(errors)), file = sys.stderr)
    sys.exit(1)

if args.annotations:
    annotations = pd.read_csv(args.annotations)
else:
    annotations = pd.DataFrame([{col.name: getattr(args, col.name) for col in AnnotationManager.INDEX_COLUMNS if not col.generated}])

am = AnnotationManager(project)
am.import_annotations(annotations)

errors, warnings = am.validate()

if len(am.errors) > 0:
    print("importation completed with {} errors and {} warnings".format(len(am.errors)+len(errors), len(warnings)), file = sys.stderr)
    print("\n".join(am.errors), file = sys.stderr)
    print("\n".join(errors), file = sys.stderr)
    print("\n".join(warnings))