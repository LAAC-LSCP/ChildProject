from ChildProject.ChildProject import ChildProject, RecordingProfile
import argparse
import sys
import os

default_profile = RecordingProfile("default")

parser = argparse.ArgumentParser(description='convert all recordings to a given audio format')
parser.add_argument("--source", help = "source data path", required = True)
parser.add_argument("--name", help = "profile name", required = True)
parser.add_argument("--format", help = "audio format (e.g. {})".format(default_profile.format), required = True)
parser.add_argument("--codec", help = "audio codec (e.g. {})".format(default_profile.codec), required = True)
parser.add_argument("--sampling", help = "sampling frequency (e.g. {})".format(default_profile.sampling), required = True)
parser.add_argument("--split", help = "split duration (e.g. 15:00:00)", required = False, default = None)

args = parser.parse_args()

profile = RecordingProfile(
    name = args.name,
    format = args.format,
    codec = args.codec,
    sampling = args.sampling,
    split = args.split
)

project = ChildProject(args.source)
results = project.convert_recordings(profile)

for error in project.errors:
    print("error: {}".format(error), file = sys.stderr)

for warning in project.warnings:
    print("warning: {}".format(warning))

if len(project.errors) > 0:
    print("conversion failed, {} error(s) occured".format(len(project.errors)), file = sys.stderr)
    print("cannot convert recordings", file = sys.stderr)
    sys.exit(1)

print("recordings successfully converted to '{}'".format(os.path.join(project.path, 'converted_recordings', profile.name)))