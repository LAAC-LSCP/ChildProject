#!/usr/bin/env python3
from ChildProject.projects import ChildProject, RecordingProfile    
from ChildProject.annotations import AnnotationManager

import argparse
import os
import pandas as pd
import sys

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

def arg(*name_or_flags, **kwargs):
    return (list(name_or_flags), kwargs)

def subcommand(args=[], parent = subparsers):
    def decorator(func):
        parser = parent.add_parser(func.__name__.replace('_', '-'), description=func.__doc__)
        for arg in args:
            parser.add_argument(*arg[0], **arg[1])
        parser.set_defaults(func=func)
    return decorator

@subcommand([
    arg("source", help = "project path"),
    arg('--ignore-files', dest='ignore_files', required = False, default = False, action = 'store_true')
])
def validate(args):
    """validate the consistency of the dataset returning detailed errors and warnings"""

    project = ChildProject(args.source)
    errors, warnings = project.validate_input_data(args.ignore_files)

    for error in errors:
        print("error: {}".format(error), file = sys.stderr)

    for warning in warnings:
        print("warning: {}".format(warning))

    if len(errors) > 0:
        print("validation failed, {} error(s) occured".format(len(errors)), file = sys.stderr)
        sys.exit(1)

@subcommand([
    arg("source", help = "project path"),
    arg("--annotations", help = "path to input annotations index (csv)", default = "")
] + [
    arg("--{}".format(col.name), help = col.description, type = str, default = None)
    for col in AnnotationManager.INDEX_COLUMNS
    if not col.generated
])
def import_annotations(args):
    """convert and import a set of annotations"""

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

@subcommand([
    arg("dataset", help = "dataset to install. Should be a valid repository name at https://github.com/LAAC-LSCP. (e.g.: solomon-data)"),
    arg("--destination", help = "destination path", required = False, default = ""),
    arg("--storage-hostname", dest = "storage_hostname", help = "ssh storage hostname (e.g. 'foberon')", required = False, default = "")
])
def import_data(args):
    """import and configures a datalad dataset"""

    import datalad.api
    import datalad.distribution.dataset

    if args.destination:
        destination = args.destination
    else:
        destination = os.path.splitext(os.path.basename(args.dataset))[0]

    datalad.api.install(source = args.dataset, path = destination)

    ds = datalad.distribution.dataset.require_dataset(
        destination,
        check_installed = True,
        purpose = 'configuration'
    )

    cmd = 'setup'
    if args.storage_hostname:
        cmd += ' "{}"'.format(args.storage_hostname)

    datalad.api.run_procedure(spec = cmd, dataset = ds)


default_profile = RecordingProfile("default")    
@subcommand([
    arg("source", help = "project path"),
    arg("--name", help = "profile name", required = True),
    arg("--format", help = "audio format (e.g. {})".format(default_profile.format), required = True),
    arg("--codec", help = "audio codec (e.g. {})".format(default_profile.codec), required = True),
    arg("--sampling", help = "sampling frequency (e.g. {})".format(default_profile.sampling), required = True),
    arg("--split", help = "split duration (e.g. 15:00:00)", required = False, default = None),
    arg('--skip-existing', dest='skip_existing', required = False, default = False, action='store_true'),
    arg('--threads', help = "amount of threads running conversions in parallel (0 = uses all available cores)", required = False, default = 0, type = int)
])
def convert(args):
    """convert recordings to a given format"""
    profile = RecordingProfile(
        name = args.name,
        format = args.format,
        codec = args.codec,
        sampling = args.sampling,
        split = args.split
    )

    project = ChildProject(args.source)
    results = project.convert_recordings(profile, skip_existing = args.skip_existing, threads = args.threads)

    for error in project.errors:
        print("error: {}".format(error), file = sys.stderr)

    for warning in project.warnings:
        print("warning: {}".format(warning))

    if len(project.errors) > 0:
        print("conversion failed, {} error(s) occured".format(len(project.errors)), file = sys.stderr)
        print("cannot convert recordings", file = sys.stderr)
        sys.exit(1)

    print("recordings successfully converted to '{}'".format(os.path.join(project.path, 'converted_recordings', profile.name)))

@subcommand([
    arg("source", help = "source data path"),
    arg("--stats", help = "stats to retrieve (comma-separated)", required = False, default = "")
])
def stats(args):
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

def main():
    args = parser.parse_args()
    args.func(args)
