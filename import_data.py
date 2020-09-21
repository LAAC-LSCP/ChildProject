from ChildProject.ChildProject import ChildProject
import argparse
import sys

parser = argparse.ArgumentParser(description='')
parser.add_argument("--source", help = "source data path", required = True)
parser.add_argument("--destination", help = "destination path", required = True)
parser.add_argument("--follow-symlinks", help = "follow symlinks", required = False, default = False)

args = parser.parse_args()

project = ChildProject(args.source)

results = project.validate_input_data()

for error in results['errors']:
    print("error: {}".format(error), file = sys.stderr)

for warning in results['warnings']:
    print("warning: {}".format(warning))

if len(results['errors']) > 0:
    print("validation failed, {} error(s) occured".format(len(results['errors'])), file = sys.stderr)
    print("cannot import data", file = sys.stderr)
    sys.exit(1)

project.import_data(args.destination, follow_symlinks = args.follow_symlinks)
print("working directory successfully created in '{}'".format(args.destination))