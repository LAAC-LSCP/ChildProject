#!/usr/bin/env python3
from .projects import ChildProject, RAW_RECORDINGS, METADATA_FOLDER, RECORDINGS_CSV, CHILDREN_CSV
from .annotations import AnnotationManager
from .pipelines.samplers import SamplerPipeline
from .pipelines.eafbuilder import EafBuilderPipeline
from .pipelines.zooniverse import ZooniversePipeline
from .pipelines.metrics import MetricsPipeline
from .pipelines.metrics import MetricsSpecificationPipeline
from .pipelines.processors import AudioProcessingPipeline
from .pipelines.anonymize import AnonymizationPipeline
from .utils import read_wav, calculate_shift, get_audio_duration
from . import __version__

from ChildProject import __name__

from .pipelines.derivations import DERIVATIONS

import argparse
import os
from pathlib import Path
import glob
import pandas as pd
import sys
import random
import logging

# add this to setup,py in the requires section and in requirements.txt
import colorlog

# Create a ColorFormatter with desired color settings
color_formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)-8s%(reset)s %(message)s %(purple)s[%(name)s]%(reset)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
)

# Create a StreamHandler and set the formatter
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(color_formatter)
#stream_handler.formatter._fmt = '%(log_color)s%(levelname)-8s%(reset)s  <%(name)s>: %(message)s'

# Create a logger and add the handlers for CLI calls
logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

# Setting up the parse of arguments
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()
parser.add_argument('--version', action='version', version="{} {}".format(__name__, __version__), help='displays the current version of the package')

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
            logger.error(
                "dataset validation failed, %d error(s) occurred. Cannot continue. Please run the validation procedure to list and correct all errors.",
                len(errors),
            )
            sys.exit(1)
        else:
            logger.warning(
                "dataset validation failed, %d error(s) occurred. Proceeding despite errors; expect failures.",
                len(errors),
            )


@subcommand(
    [
        arg("source", help="project path"),
        arg(
            "--force","-f",
            help="ignore existing files and create strcture anyway",
            action="store_true",
        ),
    ]
)
def init(args):
    path = Path(args.source)

    files = glob.glob(str(path / '*'))
    if len(files) != 0 :
        raise ValueError("Directory {} not empty, cannot create a project".format(path))

    os.makedirs(path / RAW_RECORDINGS, exist_ok=args.force)
    os.makedirs(path / METADATA_FOLDER, exist_ok=args.force)
    os.makedirs(path / 'extra', exist_ok=args.force)
    os.makedirs(path / 'scripts', exist_ok=args.force)
    os.makedirs(path / 'annotations', exist_ok=args.force)
    open(path / 'README.md', 'a').close()
    pd.DataFrame(columns = [col.name for col in ChildProject.RECORDINGS_COLUMNS if col.required]).to_csv(
        path / METADATA_FOLDER / RECORDINGS_CSV, index=False
    )
    pd.DataFrame(columns=[col.name for col in ChildProject.CHILDREN_COLUMNS if col.required]).to_csv(
        path / METADATA_FOLDER / CHILDREN_CSV, index=False
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
    
    am = AnnotationManager(project)

    errors.extend(am.errors)
    warnings.extend(am.warnings)

    if args.annotations:
        
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
        pass
        logger.error('%s',error)

    for warning in warnings:
        pass
        logger.warning('%s',warning )
    if len(errors) > 0:
        logger.warning('validation failed, %s error(s) occured', len(errors))
        sys.exit(1)

    logger.info('validation successfully completed with %d warning(s).', len(warnings))


@subcommand(
    [
        arg("source", help="project path"),
        arg(
            "--annotations",
            help="path to input annotations dataframe (csv) [only for bulk importation]",
            default="",
        ),       
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
    ] +
    [
     arg("--threads", help="amount of threads to run on", type=int, default=0),
     arg("--overwrite-existing","--ow", help="overwrites existing annotation file if should generate the same output file (useful when reimporting", action='store_true'),
     ]
)
def import_annotations(args):
    """convert and import a set of annotations"""

    project = ChildProject(args.source)

    perform_validation(project, require_success=True, ignore_recordings=True)

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
    imported, errors_imp = am.import_annotations(annotations, args.threads, overwrite_existing=args.overwrite_existing)
    
    if errors_imp is not None and errors_imp.shape[0] > 0:
        logger.error('The importation failed for %d entry/ies',errors_imp.shape[0])
        logger.debug(errors_imp)

    if imported is not None and imported.shape[0] > 0:
        errors, warnings = am.validate(imported, threads=args.threads)
 
        if len(errors) > 0:
            logger.error(
                "in the resulting annotations %s errors were found:\n%s",
                len(errors),
                "\n".join(errors),
            )


@subcommand(
    [
        arg("source", help="project path"),
        arg("derivation", help="Type of derivation", type=str, choices=DERIVATIONS.keys()),
        arg("--input-set", "-i", help="input set", required=True, type=str),
        arg("--output-set", "-o", help="output set", required=True, type=str),
        arg("--threads", help="amount of threads to run on", type=int, default=0),
        arg("--overwrite-existing", "--ow",
            help="overwrites existing annotation file when deriving (useful when reimporting), False by default",
            action='store_true'),
    ]
)
def derive_annotations(args):
    """derive a set of annotations"""

    project = ChildProject(args.source)

    perform_validation(project, require_success=True, ignore_recordings=True)

    am = AnnotationManager(project)
    imported, errors_der = am.derive_annotations(args.input_set, args.output_set, args.derivation, args.threads, overwrite_existing=args.overwrite_existing)

    if errors_der is not None and errors_der.shape[0] > 0:
        logger.error('The derivation failed for %d entry/ies', errors_der.shape[0])
        logger.debug(errors_der)

    if imported is not None and imported.shape[0] > 0:
        errors, warnings = am.validate(imported, threads=args.threads)

        if len(errors) > 0:
            logger.error(
                "in the resulting annotations %s errors were found:\n%s",
                len(errors),
                "\n".join(errors),
            )


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
    errors, warnings = project.validate(ignore_recordings=True)

    perform_validation(project, require_success=True, ignore_recordings=True)

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


@subcommand([])
def interpreter(args):
    print(sys.executable)


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

    perform_validation(project, require_success=True, ignore_recordings=True)

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

    perform_validation(project, require_success=True, ignore_recordings=True)

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

    perform_validation(project, require_success=True, ignore_recordings=True)

    am = AnnotationManager(project)
    project.read()

    output = "\n\033[1mrecordings ({:.2f} hours)\033[0m:\n".format(
        project.recordings.dropna(subset=["recording_filename"])["duration"].sum() / (3600 * 1000)
    )

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

        output += "\033[94m%s\033[0m: %s, %d/%d files locally available\n" % (
                    recording_device_type, duration, available, len(recordings))


    output += "\n\033[1mannotations\033[0m:\n"
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

        output += "\033[94m%s\033[0m: %.2f hours, %s/%s files locally available\n" % (
                    annotation_set, duration_covered / (3600 * 1000), available, len(annotations))

    logger.info(output)


@subcommand(
    [arg("source", help="source data path"), arg("variable", help="name of the variable")]
)
def explain(args):
    """prints information about a certain metadata variable"""

    variable = args.variable.lower()

    project = ChildProject(args.source)
    project.read()

    documentation = project.read_documentation()
    documentation = documentation[documentation["variable"].str.lower() == variable]

    if not len(documentation):
        documentation = [
            {
                'variable': col.name,
                'description': col.description,
                'table': 'recordings',
                'scope': 'unknown' 
            }
            for col in project.RECORDINGS_COLUMNS
        ]

        documentation += [
            {
                'variable': col.name,
                'description': col.description,
                'table': 'children',
                'scope': 'unknown' 
            }
            for col in project.CHILDREN_COLUMNS
        ]

        documentation += [
            {
                'variable': col.name,
                'description': col.description,
                'table': 'annotations',
                'scope': 'unknown' 
            }
            for col in AnnotationManager.SEGMENTS_COLUMNS
        ]

        documentation = pd.DataFrame(documentation)
        documentation = documentation[documentation["variable"].str.lower() == variable]


    if not len(documentation):
        logger.info("Could not find any documentation for variable '%s'", variable)
        return
    
    logger.info("Matching documentation for '%s':", variable)
    for doc in documentation.to_dict(orient = 'records'):
        logger.info("\n\033[94mtable\033[0m: %s", doc['table'])
        logger.info("\033[94mdescription\033[0m: %s", {doc['description']})

        if 'values' in doc and not pd.isnull(doc['values']):
            logger.info("\033[94mvalues\033[0m: %s", {doc['values']})

        if 'annotation_set' in doc and not pd.isnull(doc['annotation_set']):
            logger.info("\033[94mannotation set(s)\033[0m: %s", doc['annotation_set'])

        if 'scope' in doc and not pd.isnull(doc['scope']):
            logger.info("\033[94mscope\033[0m: %s", doc['scope'])

@subcommand(
    [
        arg("source", help="source data path"),
        arg("--profile", help="which audio profile to use", default=""),
        arg("--force", help="overwrite if column exists", action="store_true"),
    ]
)
def compute_durations(args):
    """creates a 'duration' column into metadata/recordings. duration is in ms"""
    project = ChildProject(args.source)

    #accumulate to false b/c we don't want to write confidential info into recordings.csv
    perform_validation(project, require_success=True, ignore_recordings=True, accumulate=False)

    if "duration" in project.recordings.columns:
        if not args.force:
            logger.info("The 'duration' column already exists, aborting the procces")
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
    recordings["duration"] = recordings["duration"].astype("Int64")
    
    project.recordings = recordings.copy()
    project.write_recordings()
    
@subcommand(
    [
        arg("source", help="project path"),
        arg("audio1", help="name of the first audio file as it is indexed in recordings.csv in column <recording_filename>"),
        arg("audio2", help="name of the second audio file as it is indexed in recordings.csv in column <recording_filename>"),
        arg("--profile", help="which audio profile to use", default=""),
        arg("--interval", help="duration in minutes of the window used to build the correlation score", default=5, type=int),
    ]
)
def compare_recordings(args):
    """computes the difference between 2 given audio files of the dataset. A divergence score is outputted, it is the average difference of audio signal over the considered sample (random point in the audio, fixed duration). Divergence scores lower than 0.1 indicate a strong proximity"""
    
    project = ChildProject(args.source)
    project.read()
    
    rec1 = project.recordings[project.recordings['recording_filename'] == args.audio1]
    if rec1.empty or rec1.shape[0] > 1: raise ValueError("{} was not found in the indexed recordings in metadata/recordings.csv or has multiple occurences".format(args.audio1))
    
    rec2 = project.recordings[project.recordings['recording_filename'] == args.audio2]
    if rec2.empty or rec2.shape[0] > 1: raise ValueError("{} was not found in the indexed recordings in metadata/recordings.csv or has multiple occurences".format(args.audio2))
    
    if 'duration' not in rec1.columns: 
        logger.warning("WARNING : duration was not found for audio %s. We attempt to compute it...", args.audio1)
        rec1["duration"].iloc[0] = get_audio_duration(project.get_recording_path(args.audio1, args.profile))
    if 'duration' not in rec2.columns: 
        logger.watning("WARNING : duration was not found for audio %s. We attempt to compute it...", args.audio2)
        rec2["duration"].iloc[0] = get_audio_duration(project.get_recording_path(args.audio2, args.profile))
        
    if rec1['duration'].iloc[0] != rec2['duration'].iloc[0]:
        logger.warning('WARNING : the 2 audio files have different durations, it is unlikely they are the same recording:\n%s : %dms\n%s : %dms', args.audio1, rec1['duration'].iloc[0], args.audio2, rec2['duration'].iloc[0])
 
    interval = args.interval * 60 * 1000
    
    dur = min(rec1['duration'].iloc[0],rec2['duration'].iloc[0])
    if dur < interval :
        logger.warning("WARNING : the duration of the audio is too short for an interval %dms :\nnew interval is set to %dms, this will cover the entire duration.", interval, dur)
        interval = dur
        offset = 0
    else:
        offset = random.uniform(0, dur - interval)/1000
    
    avg,size = calculate_shift(
        project.get_recording_path(rec1['recording_filename'].iloc[0],args.profile),
        project.get_recording_path(rec2['recording_filename'].iloc[0],args.profile),
        offset,
        offset,
        interval/1000
    )
    
    if size < 48000 : 
        logger.warning('WARNING : the number of values (%d) in the sample is low, raise the interval value, if possible, for a more reliable analysis', size)
    logger.info("RESULTS :\ndivergence score = %d over a sample of %d values\nREFERENCE :\ndivergence score < 0.1 => the 2 files seem very similar\ndivergence score > 1   => sizable difference", avg, size)

def main():
    register_pipeline("process", AudioProcessingPipeline)
    register_pipeline("sampler", SamplerPipeline)
    register_pipeline("zooniverse", ZooniversePipeline)
    register_pipeline("eaf-builder", EafBuilderPipeline)
    register_pipeline("anonymize", AnonymizationPipeline)
    register_pipeline("metrics", MetricsPipeline)
    register_pipeline("metrics-specification", MetricsSpecificationPipeline)

    args = parser.parse_args()
    args.func(args)
