from ChildProject.projects import ChildProject
import argparse
import sys

parser = argparse.ArgumentParser(description='Validate raw data formatting and consistency')
parser.add_argument("--source", help = "source data path", required = True)
args = parser.parse_args()

project = ChildProject(args.source)

errors, warnings = project.validate_input_data()

for error in errors:
    print("error: {}".format(error), file = sys.stderr)

for warning in warnings:
    print("warning: {}".format(warning))

if len(errors) > 0:
    print("validation failed, {} error(s) occured".format(len(errors)), file = sys.stderr)
    sys.exit(1)

