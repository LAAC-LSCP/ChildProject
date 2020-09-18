from ChildProject.ChildProject import ChildProject
import argparse
import sys

parser = argparse.ArgumentParser(description='Validate raw data formatting and consistency')
parser.add_argument("--source", help = "source data path", required = True)
args = parser.parse_args()

project = ChildProject()
project.raw_data_path = args.source

try:
    results = project.validate_input_data()
except Exception as e:
    print("Validation stopped with the following error:", file = sys.stderr)
    print(str(e), file = sys.stderr)
    sys.exit(1)

for error in results['errors']:
    print("error: {}".format(error), file = sys.stderr)

for warning in results['warnings']:
    print("warning: {}".format(warning))

if len(results['errors']) > 0:
    print("validation failed, {} error(s) occured".format(len(results['errors'])), file = sys.stderr)

