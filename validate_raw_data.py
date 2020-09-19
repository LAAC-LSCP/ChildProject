from ChildProject.ChildProject import ChildProject
import argparse
import sys

parser = argparse.ArgumentParser(description='Validate raw data formatting and consistency')
parser.add_argument("--source", help = "source data path", required = True)
args = parser.parse_args()

project = ChildProject()
project.raw_data_path = args.source

results = project.validate_input_data()

for error in results['errors']:
    print("error: {}".format(error), file = sys.stderr)

for warning in results['warnings']:
    print("warning: {}".format(warning))

if len(results['errors']) > 0:
    print("validation failed, {} error(s) occured".format(len(results['errors'])), file = sys.stderr)
    sys.exit(1)

