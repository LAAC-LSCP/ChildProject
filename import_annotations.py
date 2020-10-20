from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
import argparse
import pandas as pd
import sys

parser = argparse.ArgumentParser(description='')
parser.add_argument("--source", help = "project path", required = True)
parser.add_argument("--annotations", help = "path to input annotations index (csv)", required = True)

args = parser.parse_args()

project = ChildProject(args.source)
errors, warnings = project.validate_input_data()

if len(errors) > 0:
    print("validation failed, {} error(s) occured".format(len(errors)), file = sys.stderr)
    sys.exit(1)

am = AnnotationManager(project)
am.import_annotations(pd.read_csv(args.annotations))

if len(am.errors) > 0:
    print("importation completed with {} errors".format(len(am.errors)), file = sys.stderr)
    print("\n".join(am.errors), file = sys.stderr)