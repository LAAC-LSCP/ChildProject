#!/usr/bin/env python3
from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from ChildProject.pipelines import *

import argparse
import os
import pandas as pd
import sys

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()


def arg(*name_or_flags, **kwargs):
    return (list(name_or_flags), kwargs)


def get_doc_summary(doc):
    return doc.split("\n")[0] if doc else ""


def subcommand(args=[], parent=subparsers):
    def decorator(func):
        parser = parent.add_parser(
            func.__name__.replace("_", "-"), description=func.__doc__
        )
        for arg in args:
            parser.add_argument(*arg[0], **arg[1])
        parser.set_defaults(func=func)

    return decorator


def register_pipeline(subcommand, cls):
    _parser = subparsers.add_parser(
        subcommand, description=get_doc_summary(cls.run.__doc__)
    )
    cls.setup_parser(_parser)
    _parser.set_defaults(func=lambda args: cls().run(**vars(args)))


def perform_validation(project: ChildProject, require_success: bool = True, **args):
    errors, warnings = project.validate(**args)

    if len(errors) > 0:
        if require_success:
            print(
                "[\033[31merror\033[0m]: dataset validation failed, {} error(s) occured.\nCannot continue. Please run the validation procedure to list and correct all errors.".format(
                    len(errors)
                ),
                file=sys.stderr,
            )
            sys.exit(1)
        else:
            print(
                "[\033[33mwarning\033[0m]: dataset validation failed, {} error(s) occured.\nProceeding despite errors; expect failures.".format(
                    len(errors)
                )
            )


@subcommand(
    [
        arg("source", help="project path"),
        arg(
            "--ignore-recordings",
            help="ignore missing audio files",
            dest="ignore_recordings",
            required=False,
            default=False,
            action="store_true",
        ),
        arg(
            "--profile",
            help="which recording profile to validate",
            dest="profile",
            required=False,
            default=None,
        ),
        arg(
            "--annotations",
            help="path to or name of each annotation set(s) to check (e.g. 'vtc' or '/path/to/dataset/annotations/vtc')",
            dest="annotations",
            required=False,
            default=[],
            nargs="+",
        ),
        arg(
            "--threads",
            help="amount of threads to run on (only applies to --annotations)",
            type=int,
            default=0,
        ),
    ]
)
def validate(args):
    """validate the consistency of the dataset returning detailed errors and warnings"""

    project = ChildProject(args.source)
    errors, warnings = project.validate(args.ignore_recordings, args.profile)

    if args.annotations:
        am = AnnotationManager(project)

        errors.extend(am.errors)
        warnings.extend(am.warnings)

        annotations = am.annotations

        if all(map(lambda x: os.path.exists(x) or os.path.islink(x), args.annotations)):
            args.annotations = {am.set_from_path(set) for set in args.annotations} - {
                None
            }

        sets = list(args.annotations) + sum(
            [am.get_subsets(s, recursive=True) for s in args.annotations], []
        )
        sets = set(sets)

        if not sets.issubset(set(annotations["set"].unique())):
            missing_sets = sets - set(annotations["set"].unique())
            errors.append(
                "the following annotation sets are not indexed: {}".format(
                    ",".join(missing_sets)
                )
            )

        annotations = annotations[annotations["set"].isin(sets)]

        annotations_errors, annotations_warnings = am.validate(
            annotations=annotations, threads=args.threads
        )
        errors.extend(annotations_errors)
        warnings.extend(annotations_warnings)

    for error in errors:
        print("error: {}".format(error))

    for warning in warnings:
        print("warning: {}".format(warning))

    if len(errors) > 0:
        print("validation failed, {} error(s) occured".format(len(errors)))
        sys.exit(1)

    print("validation successfully completed with {} warning(s).".format(len(warnings)))


@subcommand(
    [
        arg("source", help="project path"),
        arg(
            "--annotations",
            help="path to input annotations dataframe (csv) [only for bulk importation]",
            default="",
        ),
        arg("--threads", help="amount of threads to run on", type=int, default=0),
    ]
    + [
        arg(
            "--{}".format(col.name),
            help=col.description,
            type=str,
            default=None,
            choices=col.choices if col.choices else None,
        )
        for col in AnnotationManager.INDEX_COLUMNS
        if not col.generated
    ]
)
def import_annotations(args):
    """convert and import a set of annotations"""

    project = ChildProject(args.source)

    perform_validation(project, require_success=True, ignore_files=True)

    if args.annotations:
        annotations = pd.read_csv(args.annotations)
    else:
        annotations = pd.DataFrame(
            [
                {
                    col.name: getattr(args, col.name)
                    for col in AnnotationManager.INDEX_COLUMNS
                    if not col.generated
                }
            ]
        )

    am = AnnotationManager(project)
    imported = am.import_annotations(annotations, args.threads)

    errors, warnings = am.validate(annotations=imported, threads=args.threads)

    if len(am.errors) > 0:
        print(
            "importation completed with {} errors and {} warnings".format(
                len(am.errors) + len(errors), len(warnings)
            ),
            file=sys.stderr,
        )
        print("\n".join(am.errors), file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        print("\n".join(warnings))


@subcommand(
    [
        arg("source", help="project path"),
        arg("--left-set", help="left set", required=True),
        arg("--right-set", help="right set", required=True),
        arg(
            "--left-columns",
            help="comma-separated columns to merge from the left set",
            required=True,
        ),
        arg(
            "--right-columns",
            help="comma-separated columns to merge from the right set",
            required=True,
        ),
        arg("--output-set", help="name of the output set", required=True),
        arg(
            "--threads",
            help="amount of threads to run on (default: 1)",
            type=int,
            default=1,
        ),
    ]
)
def merge_annotations(args):
    """merge segments sharing identical onset and offset from two sets of annotations"""
    project = ChildProject(args.source)
    errors, warnings = project.validate(ignore_files=True)

    perform_validation(project, require_success=True, ignore_files=True)

    am = AnnotationManager(project)
    am.read()
    am.merge_sets(
        left_set=args.left_set,
        right_set=args.right_set,
        left_columns=args.left_columns.split(","),
        right_columns=args.right_columns.split(","),
        output_set=args.output_set,
        threads=args.threads,
    )


@subcommand(
    [
        arg("source", help="project path"),
        arg("--destination", help="output CSV dataframe destination", required=True),
        arg("--sets", help="annotation sets to intersect", nargs="+", required=True),
        arg(
            "--annotations",
            help="path a custom input CSV dataframe of annotations to intersect. By default, the whole index of the project will be used.",
            default=None,
        ),
    ]
)
def intersect_annotations(args):
    """calculate the intersection of the annotations belonging to the given sets"""

    if args.annotations:
        annotations = pd.read_csv(args.annotations)
    else:
        project = ChildProject(args.source)
        am = AnnotationManager(project)
        am.read()
        annotations = am.annotations

    intersection = AnnotationManager.intersection(annotations, args.sets)
    intersection.to_csv(args.destination, index=False)


@subcommand(
    [
        arg("source", help="project path"),
        arg("--set", help="set to remove", required=True),
        arg("--recursive", help="enable recursive mode", action="store_true"),
    ]
)
def remove_annotations(args):
    """remove converted annotations of a given set and their entries in the index"""
    project = ChildProject(args.source)

    perform_validation(project, require_success=True, ignore_files=True)

    am = AnnotationManager(project)
    am.read()
    am.remove_set(args.set, recursive=args.recursive)


@subcommand(
    [
        arg("source", help="project path"),
        arg("--set", help="set to rename", required=True),
        arg("--new-set", help="new name for the set", required=True),
        arg("--recursive", help="enable recursive mode", action="store_true"),
        arg("--ignore-errors", help="proceed despite errors", action="store_true"),
    ]
)
def rename_annotations(args):
    """rename a set of annotations by moving the files and updating the index accordingly"""

    project = ChildProject(args.source)

    perform_validation(project, require_success=True, ignore_files=True)

    am = AnnotationManager(project)
    am.read()
    am.rename_set(
        args.set,
        args.new_set,
        recursive=args.recursive,
        ignore_errors=args.ignore_errors,
    )


@subcommand([arg("source", help="source data path")])
def overview(args):
    """prints an overview of the contents of a given dataset"""

    project = ChildProject(args.source)

    perform_validation(project, require_success=True, ignore_files=True)

    am = AnnotationManager(project)
    project.read()

    print("\n\033[1mrecordings\033[0m:")
    _recordings = (
        project.recordings.dropna(subset=["recording_filename"])
        .sort_values(["recording_device_type", "date_iso"])
        .groupby("recording_device_type")
    )

    for recording_device_type, recordings in _recordings:
        if "duration" in recordings.columns:
            duration = "{:.2f} hours".format(
                recordings["duration"].sum() / (3600 * 1000)
            )
        else:
            duration = "unknown duration"

        available = (
            recordings["recording_filename"]
            .apply(
                lambda recording_filename: 1
                if os.path.exists(
                    os.path.join(project.path, "recordings", "raw", recording_filename)
                )
                else 0
            )
            .sum()
        )

        print(
            "\033[94m{}\033[0m: {}, {}/{} files locally available".format(
                recording_device_type, duration, available, len(recordings)
            )
        )

    print("\n\033[1mannotations\033[0m:")
    _annotations = (
        am.annotations.dropna(subset=["annotation_filename"])
        .sort_values(["set", "imported_at"])
        .drop_duplicates(["set", "annotation_filename"], keep="last")
        .groupby("set")
    )

    for annotation_set, annotations in _annotations:
        duration_covered = (
            annotations["range_offset"].sum() - annotations["range_onset"].sum()
        )
        available = (
            annotations["annotation_filename"]
            .apply(
                lambda annotation_filename: 1
                if os.path.exists(
                    os.path.join(
                        project.path,
                        "annotations",
                        annotation_set,
                        "converted",
                        annotation_filename,
                    )
                )
                else 0
            )
            .sum()
        )

        print(
            "\033[94m{}\033[0m: {:.2f} hours, {}/{} files locally available".format(
                annotation_set,
                duration_covered / (3600 * 1000),
                available,
                len(annotations),
            )
        )


@subcommand(
    [
        arg("source", help="source data path"),
        arg("--profile", help="which audio profile to use", default=""),
        arg("--force", help="overwrite if column exists", action="store_true"),
    ]
)
def compute_durations(args):
    """creates a 'duration' column into metadata/recordings"""
    project = ChildProject(args.source)

    perform_validation(project, require_success=True, ignore_files=True)

    if "duration" in project.recordings.columns:
        if not args.force:
            print("duration exists, aborting")
            return

        project.recordings.drop(columns=["duration"], inplace=True)

    durations = project.compute_recordings_duration(profile=args.profile).dropna()

    recordings = project.recordings.merge(
        durations[durations["recording_filename"] != "NA"],
        how="left",
        left_on="recording_filename",
        right_on="recording_filename",
    )
    recordings["duration"].fillna(0, inplace=True)
    recordings["duration"] = recordings["duration"].astype(int)
    recordings.to_csv(
        os.path.join(project.path, "metadata/recordings.csv"), index=False
    )


def main():
    register_pipeline("process", AudioProcessingPipeline)
    register_pipeline("sampler", SamplerPipeline)
    register_pipeline("zooniverse", ZooniversePipeline)
    register_pipeline("eaf-builder", EafBuilderPipeline)
    register_pipeline("anonymize", AnonymizationPipeline)
    register_pipeline("metrics", MetricsPipeline)

    args = parser.parse_args()
    args.func(args)
