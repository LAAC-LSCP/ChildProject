import datetime
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from functools import reduce, partial
from shutil import move, rmtree
import sys
import traceback
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

from . import __version__
from .projects import ChildProject
from .converters import *
from .tables import IndexTable, IndexColumn
from .utils import Segment, intersect_ranges, path_is_parent


class AnnotationManager:
    INDEX_COLUMNS = [
        IndexColumn(
            name="set",
            description="name of the annotation set (e.g. VTC, annotator1, etc.)",
            required=True,
        ),
        IndexColumn(
            name="recording_filename",
            description="recording filename as specified in the recordings index",
            required=True,
        ),
        IndexColumn(
            name="time_seek",
            description="shift between the timestamps in the raw input annotations and the actual corresponding timestamps in the recordings (in milliseconds)",
            regex=r"(\-?)([0-9]+)",
            required=True,
        ),
        IndexColumn(
            name="range_onset",
            description="covered range onset timestamp in milliseconds (since the start of the recording)",
            regex=r"[0-9]+",
            required=True,
        ),
        IndexColumn(
            name="range_offset",
            description="covered range offset timestamp in milliseconds (since the start of the recording)",
            regex=r"[0-9]+",
            required=True,
        ),
        IndexColumn(
            name="raw_filename",
            description="annotation input filename location, relative to `annotations/<set>/raw`",
            filename=True,
            required=True,
        ),
        IndexColumn(
            name="format",
            description="input annotation format",
            choices=[*converters.keys(), "NA"],
            required=False,
        ),
        IndexColumn(
            name="filter",
            description="source file to filter in (for rttm and alice only)",
            required=False,
        ),
        IndexColumn(
            name="annotation_filename",
            description="output formatted annotation location, relative to `annotations/<set>/converted (automatic column, don't specify)",
            filename=True,
            required=False,
            generated=True,
        ),
        IndexColumn(
            name="imported_at",
            description="importation date (automatic column, don't specify)",
            datetime="%Y-%m-%d %H:%M:%S",
            required=False,
            generated=True,
        ),
        IndexColumn(
            name="package_version",
            description="version of the package used when the importation was performed",
            regex=r"[0-9]+\.[0-9]+\.[0-9]+",
            required=False,
            generated=True,
        ),
        IndexColumn(
            name="error",
            description="error message in case the annotation could not be imported",
            required=False,
            generated=True,
        ),
    ]

    SEGMENTS_COLUMNS = [
        IndexColumn(
            name="raw_filename",
            description="raw annotation path relative, relative to `annotations/<set>/raw`",
            required=True,
        ),
        IndexColumn(
            name="segment_onset",
            description="segment onset timestamp in milliseconds (since the start of the recording)",
            regex=r"([0-9]+)",
            required=True,
        ),
        IndexColumn(
            name="segment_offset",
            description="segment end time in milliseconds (since the start of the recording)",
            regex=r"([0-9]+)",
            required=True,
        ),
        IndexColumn(
            name="speaker_id", description="identity of speaker in the annotation"
        ),
        IndexColumn(
            name="speaker_type",
            description="class of speaker (FEM = female adult, MAL = male adult, CHI = key child, OCH = other child)",
            choices=["FEM", "MAL", "CHI", "OCH", "NA"],
        ),
        IndexColumn(
            name="ling_type",
            description="1 if the vocalization contains at least a vowel (ie canonical or non-canonical), 0 if crying or laughing",
            choices=["1", "0", "NA"],
        ),
        IndexColumn(
            name="vcm_type",
            description="vocal maturity defined as: C (canonical), N (non-canonical), Y (crying) L (laughing), J (junk)",
            choices=["C", "N", "Y", "L", "J", "NA"],
        ),
        IndexColumn(
            name="lex_type",
            description="W if meaningful, 0 otherwise",
            choices=["W", "0", "NA"],
        ),
        IndexColumn(
            name="mwu_type",
            description="M if multiword, 1 if single word -- only filled if lex_type==W",
            choices=["M", "1", "NA"],
        ),
        IndexColumn(
            name="addressee",
            description="T if target-child-directed, C if other-child-directed, A if adult-directed, U if uncertain or other. Multiple values should be sorted and separated by commas",
            choices=["T", "C", "A", "U", "NA"],
        ),
        IndexColumn(
            name="transcription", description="orthographic transcription of the speach"
        ),
        IndexColumn(
            name="phonemes", description="amount of phonemes", regex=r"(\d+(\.\d+)?)"
        ),
        IndexColumn(
            name="syllables", description="amount of syllables", regex=r"(\d+(\.\d+)?)"
        ),
        IndexColumn(
            name="words", description="amount of words", regex=r"(\d+(\.\d+)?)"
        ),
        IndexColumn(
            name="lena_block_type",
            description="whether regarded as part as a pause or a conversation by LENA",
            choices=[
                "pause",
                "CM",
                "CIC",
                "CIOCX",
                "CIOCAX",
                "AMF",
                "AICF",
                "AIOCF",
                "AIOCCXF",
                "AMM",
                "AICM",
                "AIOCM",
                "AIOCCXM",
                "XM",
                "XIOCC",
                "XIOCA",
                "XIC",
                "XIOCAC",
            ],
        ),
        IndexColumn(
            name="lena_block_number",
            description="number of the LENA pause/conversation the segment belongs to",
            regex=r"(\d+(\.\d+)?)",
        ),
        IndexColumn(
            name="lena_conv_status",
            description="LENA conversation status",
            choices=["BC", "RC", "EC"],
        ),
        IndexColumn(
            name="lena_response_count",
            description="LENA turn count within block",
            regex=r"(\d+(\.\d+)?)",
        ),
        IndexColumn(
            name="lena_conv_floor_type",
            description="(FI): Floor Initiation, (FH): Floor Holding",
            choices=["FI", "FH"],
        ),
        IndexColumn(
            name="lena_conv_turn_type",
            description="LENA turn type",
            choices=["TIFI", "TIMI", "TIFR", "TIMR", "TIFE", "TIME", "NT"],
        ),
        IndexColumn(
            name="utterances_count",
            description="utterances count",
            regex=r"(\d+(\.\d+)?)",
        ),
        IndexColumn(
            name="utterances_length", description="utterances length", regex=r"([0-9]+)"
        ),
        IndexColumn(
            name="non_speech_length", description="non-speech length", regex=r"([0-9]+)"
        ),
        IndexColumn(
            name="average_db",
            description="average dB level",
            regex=r"(\-?)(\d+(\.\d+)?)",
        ),
        IndexColumn(
            name="peak_db", description="peak dB level", regex=r"(\-?)(\d+(\.\d+)?)"
        ),
        IndexColumn(
            name="child_cry_vfx_len", description="childCryVfxLen", regex=r"([0-9]+)"
        ),
        IndexColumn(name="utterances", description="LENA utterances details (json)"),
        IndexColumn(name="cries", description="cries (json)"),
        IndexColumn(name="vfxs", description="Vfx (json)"),
    ]

    def __init__(self, project: ChildProject):
        """AnnotationManager constructor

        :param project: :class:`ChildProject` instance of the target dataset.
        :type project: :class:`ChildProject`
        """
        self.project = project
        self.annotations = None
        self.errors = []

        if not isinstance(project, ChildProject):
            raise ValueError("project should derive from ChildProject")

        self.project.read()

        index_path = os.path.join(self.project.path, "metadata/annotations.csv")
        if not os.path.exists(index_path):
            open(index_path, "w+").write(",".join([c.name for c in self.INDEX_COLUMNS]))

        self.errors, self.warnings = self.read()

    def read(self) -> Tuple[List[str], List[str]]:
        """Read the index of annotations from ``metadata/annotations.csv`` and store it into
        self.annotations.


        :return: a tuple containing the list of errors and the list of warnings generated while reading the index
        :rtype: Tuple[List[str],List[str]]
        """
        table = IndexTable(
            "input",
            path=os.path.join(self.project.path, "metadata/annotations.csv"),
            columns=self.INDEX_COLUMNS,
        )
        self.annotations = table.read()
        errors, warnings = table.validate()

        duplicates = self.annotations.groupby(["set", "annotation_filename"]).agg(
            count=("range_offset", "count")
        )
        duplicates = duplicates[duplicates["count"] > 1].reset_index()

        if len(duplicates):
            errors.extend(
                [
                    "duplicate reference to annotations/{}/converted/{} (appears {} times)".format(
                        dup["set"], dup["annotation_filename"], dup["count"]
                    )
                    for dup in duplicates.to_dict(orient="records")
                ]
            )

        return errors, warnings

    def validate_annotation(self, annotation: dict) -> Tuple[List[str], List[str]]:
        print(
            "validating {} from {}...".format(
                annotation["annotation_filename"], annotation["set"]
            )
        )

        segments = IndexTable(
            "segments",
            path=os.path.join(
                self.project.path,
                "annotations",
                annotation["set"],
                "converted",
                str(annotation["annotation_filename"]),
            ),
            columns=self.SEGMENTS_COLUMNS,
        )

        try:
            segments.read()
        except Exception as e:
            error_message = "error while trying to read {} from {}:\n\t{}".format(
                annotation["annotation_filename"], annotation["set"], str(e)
            )
            return [error_message], []

        return segments.validate()

    def validate(
        self, annotations: pd.DataFrame = None, threads: int = 0
    ) -> Tuple[List[str], List[str]]:
        """check all indexed annotations for errors

        :param annotations: annotations to validate, defaults to None. If None, the whole index will be scanned.
        :type annotations: pd.DataFrame, optional
        :param threads: how many threads to run the tests with, defaults to 0. If <= 0, all available CPU cores will be used.
        :type threads: int, optional
        :return: a tuple containg the list of errors and the list of warnings detected
        :rtype: Tuple[List[str], List[str]]
        """
        if not isinstance(annotations, pd.DataFrame):
            annotations = self.annotations

        annotations = annotations.dropna(subset=["annotation_filename"])

        errors, warnings = [], []

        with mp.Pool(processes=threads if threads > 0 else mp.cpu_count()) as pool:
            res = pool.map(
                self.validate_annotation, annotations.to_dict(orient="records")
            )

        errors = reduce(lambda x, y: x + y[0], res, [])
        warnings = reduce(lambda x, y: x + y[1], res, [])

        return errors, warnings

    def write(self):
        """Update the annotations index,
        while enforcing its good shape.
        """
        self.annotations[["time_seek", "range_onset", "range_offset"]].fillna(
            0, inplace=True
        )
        self.annotations[
            ["time_seek", "range_onset", "range_offset"]
        ] = self.annotations[["time_seek", "range_onset", "range_offset"]].astype(int)
        self.annotations.to_csv(
            os.path.join(self.project.path, "metadata/annotations.csv"), index=False
        )

    def _import_annotation(
        self, import_function: Callable[[str], pd.DataFrame], annotation: dict
    ):
        """import and convert ``annotation``. This function should not be called outside of this class.

        :param import_function: If callable, ``import_function`` will be called to convert the input annotation into a dataframe. Otherwise, the conversion will be performed by a built-in function.
        :type import_function: Callable[[str], pd.DataFrame]
        :param annotation: input annotation dictionary (attributes defined according to :ref:`ChildProject.annotations.AnnotationManager.SEGMENTS_COLUMNS`)
        :type annotation: dict
        :return: output annotation dictionary (attributes defined according to :ref:`ChildProject.annotations.AnnotationManager.SEGMENTS_COLUMNS`)
        :rtype: dict
        """

        source_recording = os.path.splitext(annotation["recording_filename"])[0]
        annotation_filename = "{}_{}_{}.csv".format(
            source_recording, annotation["range_onset"], annotation["range_offset"]
        )
        output_filename = os.path.join(
            "annotations", annotation["set"], "converted", annotation_filename
        )

        path = os.path.join(
            self.project.path,
            "annotations",
            annotation["set"],
            "raw",
            annotation["raw_filename"],
        )
        annotation_format = annotation["format"]

        df = None
        filter = (
            annotation["filter"]
            if "filter" in annotation and not pd.isnull(annotation["filter"])
            else None
        )

        try:
            if callable(import_function):
                df = import_function(path)
            elif annotation_format in converters:
                converter = converters[annotation_format]
                df = converter.convert(path, filter)
            else:
                raise ValueError(
                    "file format '{}' unknown for '{}'".format(annotation_format, path)
                )
        except:
            annotation["error"] = traceback.format_exc()
            print(
                "an error occured while processing '{}'".format(path), file=sys.stderr
            )
            print(traceback.format_exc(), file=sys.stderr)

        if df is None or not isinstance(df, pd.DataFrame):
            return annotation

        if not df.shape[1]:
            df = pd.DataFrame(columns=[c.name for c in self.SEGMENTS_COLUMNS])

        df["raw_filename"] = annotation["raw_filename"]

        df["segment_onset"] += int(annotation["time_seek"])
        df["segment_offset"] += int(annotation["time_seek"])
        df["segment_onset"] = df["segment_onset"].astype(int)
        df["segment_offset"] = df["segment_offset"].astype(int)

        annotation["time_seek"] = int(annotation["time_seek"])
        annotation["range_onset"] = int(annotation["range_onset"])
        annotation["range_offset"] = int(annotation["range_offset"])

        df = AnnotationManager.clip_segments(
            df, annotation["range_onset"], annotation["range_offset"]
        )

        sort_columns = ["segment_onset", "segment_offset"]
        if "speaker_type" in df.columns:
            sort_columns.append("speaker_type")

        df.sort_values(sort_columns, inplace=True)

        os.makedirs(
            os.path.dirname(os.path.join(self.project.path, output_filename)),
            exist_ok=True,
        )
        df.to_csv(os.path.join(self.project.path, output_filename), index=False)

        annotation["annotation_filename"] = annotation_filename
        annotation["imported_at"] = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        annotation["package_version"] = __version__

        if pd.isnull(annotation["format"]):
            annotation["format"] = "NA"

        return annotation

    def import_annotations(
        self,
        input: pd.DataFrame,
        threads: int = -1,
        import_function: Callable[[str], pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Import and convert annotations.

        :param input: dataframe of all annotations to import, as described in :ref:`format-input-annotations`.
        :type input: pd.DataFrame
        :param threads: If > 1, conversions will be run on ``threads`` threads, defaults to -1
        :type threads: int, optional
        :param import_function: If specified, the custom ``import_function`` function will be used to convert all ``input`` annotations, defaults to None
        :type import_function: Callable[[str], pd.DataFrame], optional
        :return: dataframe of imported annotations, as in :ref:`format-annotations`.
        :rtype: pd.DataFrame
        """

        required_columns = {
            c.name
            for c in AnnotationManager.INDEX_COLUMNS
            if c.required and not c.generated
        }
        missing_columns = required_columns - set(input.columns)

        if len(missing_columns):
            raise IndexError(
                "import_annotations requires the following missing columns: {}".format(
                    ",".join(missing_columns)
                )
            )

        missing_recordings = input[
            ~input["recording_filename"].isin(
                self.project.recordings["recording_filename"]
            )
        ]
        missing_recordings = missing_recordings["recording_filename"]

        if len(missing_recordings) > 0:
            raise ValueError(
                "cannot import annotations, because the following recordings are not referenced in the metadata:\n{}".format(
                    "\n".join(missing_recordings)
                )
            )

        input["range_onset"] = input["range_onset"].astype(int)
        input["range_offset"] = input["range_offset"].astype(int)

        builtin = input[input["format"].isin(converters.keys())]
        if not builtin["format"].map(lambda f: converters[f].THREAD_SAFE).all():
            print(
                "warning: some of the converters do not support multithread importation; running on 1 thread"
            )
            threads = 1

        if threads == 1:
            imported = input.apply(
                partial(self._import_annotation, import_function), axis=1
            ).to_dict(orient="records")
        else:
            with mp.Pool(processes=threads if threads > 0 else mp.cpu_count()) as pool:
                imported = pool.map(
                    partial(self._import_annotation, import_function),
                    input.to_dict(orient="records"),
                )

        imported = pd.DataFrame(imported)
        imported.drop(
            list(set(imported.columns) - {c.name for c in self.INDEX_COLUMNS}),
            axis=1,
            inplace=True,
        )

        self.read()
        self.annotations = pd.concat([self.annotations, imported], sort=False)
        self.write()

        return imported

    def get_subsets(self, annotation_set: str, recursive: bool = False) -> List[str]:
        """Retrieve the list of subsets belonging to a given set of annotations.

        :param annotation_set: input set
        :type annotation_set: str
        :param recursive: If True, get subsets recursively, defaults to False
        :type recursive: bool, optional
        :return: the list of subsets names
        :rtype: list
        """
        subsets = []

        path = os.path.join(self.project.path, "annotations", annotation_set)
        candidates = list(set(os.listdir(path)) - {"raw", "converted"})
        for candidate in candidates:
            subset = os.path.join(annotation_set, candidate)

            if not os.path.isdir(
                os.path.join(self.project.path, "annotations", subset)
            ):
                continue

            subsets.append(subset)

            if recursive:
                subsets.extend(self.get_subsets(subset))

        return subsets

    def remove_set(self, annotation_set: str, recursive: bool = False):
        """Remove a set of annotations, deleting every converted file and removing
        them from the index. This preserves raw annotations.

        :param annotation_set: set of annotations to remove
        :type annotation_set: str
        :param recursive: remove subsets as well, defaults to False
        :type recursive: bool, optional
        """
        self.read()

        subsets = []
        if recursive:
            subsets = self.get_subsets(annotation_set, recursive=False)

        for subset in subsets:
            self.remove_set(subset, recursive=recursive)

        path = os.path.join(
            self.project.path, "annotations", annotation_set, "converted"
        )

        try:
            rmtree(path)
        except:
            print("could not delete '{}', as it does not exist (yet?)".format(path))
            pass

        self.annotations = self.annotations[self.annotations["set"] != annotation_set]
        self.write()

    def rename_set(
        self,
        annotation_set: str,
        new_set: str,
        recursive: bool = False,
        ignore_errors: bool = False,
    ):
        """Rename a set of annotations, moving all related files
        and updating the index accordingly.

        :param annotation_set: name of the set to rename
        :type annotation_set: str
        :param new_set: new set name
        :type new_set: str
        :param recursive: rename subsets as well, defaults to False
        :type recursive: bool, optional
        :param ignore_errors: If True, keep going even if unindexed files are detected, defaults to False
        :type ignore_errors: bool, optional
        """
        self.read()

        annotation_set = annotation_set.rstrip("/").rstrip("\\")
        new_set = new_set.rstrip("/").rstrip("\\")

        current_path = os.path.join(self.project.path, "annotations", annotation_set)
        new_path = os.path.join(self.project.path, "annotations", new_set)

        if not os.path.exists(current_path):
            raise Exception("'{}' does not exists, aborting".format(current_path))

        if os.path.exists(new_path):
            raise Exception("'{}' already exists, aborting".format(new_path))

        if (
            self.annotations[self.annotations["set"] == annotation_set].shape[0] == 0
            and not ignore_errors
            and not recursive
        ):
            raise Exception(
                "set '{}' have no indexed annotation, aborting. use --ignore_errors to force"
            )

        subsets = []
        if recursive:
            subsets = self.get_subsets(annotation_set, recursive=False)

        for subset in subsets:
            self.rename_set(
                annotation_set=subset,
                new_set=re.sub(
                    r"^{}/".format(re.escape(annotation_set)),
                    os.path.join(new_set, ""),
                    subset,
                ),
                recursive=recursive,
                ignore_errors=ignore_errors,
            )

        os.makedirs(new_path, exist_ok=True)

        if os.path.exists(os.path.join(current_path, "raw")):
            move(os.path.join(current_path, "raw"), os.path.join(new_path, "raw"))

        if os.path.exists(os.path.join(current_path, "converted")):
            move(
                os.path.join(current_path, "converted"),
                os.path.join(new_path, "converted"),
            )

        self.annotations.loc[
            (self.annotations["set"] == annotation_set), "set"
        ] = new_set
        self.write()

    def merge_annotations(
        self, left_columns, right_columns, columns, output_set, input
    ):
        left_annotations = input["left_annotations"]
        right_annotations = input["right_annotations"]

        annotations = left_annotations.copy()
        annotations["format"] = ""
        annotations["annotation_filename"] = annotations.apply(
            lambda annotation: "{}_{}_{}.csv".format(
                os.path.splitext(annotation["recording_filename"])[0],
                annotation["range_onset"],
                annotation["range_offset"],
            ),
            axis=1,
        )

        for key in columns:
            annotations[key] = columns[key]

        annotations["set"] = output_set

        left_annotation_files = [
            os.path.join(
                self.project.path,
                "annotations",
                a["set"],
                "converted",
                a["annotation_filename"],
            )
            for a in left_annotations.to_dict(orient="records")
        ]
        left_missing_annotations = [
            f for f in left_annotation_files if not os.path.exists(f)
        ]

        right_annotation_files = [
            os.path.join(
                self.project.path,
                "annotations",
                a["set"],
                "converted",
                a["annotation_filename"],
            )
            for a in right_annotations.to_dict(orient="records")
        ]
        right_missing_annotations = [
            f for f in right_annotation_files if not os.path.exists(f)
        ]

        if left_missing_annotations:
            raise Exception(
                "the following annotations from the left set are missing: {}".format(
                    ",".join(left_missing_annotations)
                )
            )

        if right_missing_annotations:
            raise Exception(
                "the following annotations from the right set are missing: {}".format(
                    ",".join(right_missing_annotations)
                )
            )

        left_segments = self.get_segments(left_annotations)
        right_segments = self.get_segments(right_annotations)

        merge_columns = ["interval", "segment_onset", "segment_offset"]

        lc = merge_columns + left_columns + ["raw_filename", "time_seek"]
        rc = merge_columns + right_columns + ["raw_filename"]

        left_segments = left_segments.reindex(
            left_segments.columns.union(lc, sort=False), axis=1, fill_value="NA"
        )
        right_segments = right_segments.reindex(
            right_segments.columns.union(rc, sort=False), axis=1, fill_value="NA"
        )

        output_segments = left_segments[list(lc)].merge(
            right_segments[list(rc)],
            how="outer",
            left_on=merge_columns,
            right_on=merge_columns,
        )

        output_segments["segment_onset"] = (
            output_segments["segment_onset"].fillna(0).astype(int)
        )
        output_segments["segment_offset"] = (
            output_segments["segment_offset"].fillna(0).astype(int)
        )

        output_segments["raw_filename"] = (
            output_segments["raw_filename_x"] + "," + output_segments["raw_filename_y"]
        )

        annotations.drop(columns="raw_filename", inplace=True)
        annotations = annotations.merge(
            output_segments[["interval", "raw_filename"]].dropna().drop_duplicates(),
            how="left",
            left_on="interval",
            right_on="interval",
        )
        annotations.rename(columns={"raw_filename": "raw_filename"}, inplace=True)
        annotations["generated_at"] = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        output_segments["raw_filename"] = (
            output_segments["raw_filename_x"].fillna("")
            + ","
            + output_segments["raw_filename_y"].fillna("")
        )
        output_segments.drop(
            columns=["raw_filename_x", "raw_filename_y", "time_seek"], inplace=True
        )

        output_segments.fillna("NA", inplace=True)

        for annotation in annotations.to_dict(orient="records"):
            interval = annotation["interval"]
            annotation_filename = annotation["annotation_filename"]
            annotation_set = annotation["set"]

            os.makedirs(
                os.path.dirname(
                    os.path.join(
                        self.project.path,
                        "annotations",
                        annotation_set,
                        "converted",
                        annotation_filename,
                    )
                ),
                exist_ok=True,
            )

            segments = output_segments[output_segments["interval"] == interval]
            segments.drop(
                columns=list(
                    set(segments.columns) - {c.name for c in self.SEGMENTS_COLUMNS}
                ),
                inplace=True,
            )
            segments.to_csv(
                os.path.join(
                    self.project.path,
                    "annotations",
                    annotation_set,
                    "converted",
                    annotation_filename,
                ),
                index=False,
            )

        return annotations

    def merge_sets(
        self,
        left_set: str,
        right_set: str,
        left_columns: List[str],
        right_columns: List[str],
        output_set: str,
        columns: dict = {},
        threads=-1,
    ):
        """Merge columns from ``left_set`` and ``right_set`` annotations, 
        for all matching segments, into a new set of annotations named
        ``output_set``.

        :param left_set: Left set of annotations.
        :type left_set: str
        :param right_set: Right set of annotations.
        :type right_set: str
        :param left_columns: Columns which values will be based on the left set.
        :type left_columns: List
        :param right_columns: Columns which values will be based on the right set.
        :type right_columns: List
        :param output_set: Name of the output annotations set.
        :type output_set: str
        :return: [description]
        :rtype: [type]
        """
        assert left_set != right_set, "sets must differ"
        assert not (
            set(left_columns) & set(right_columns)
        ), "left_columns and right_columns must be disjoint"

        union = set(left_columns) | set(right_columns)
        all_columns = {c.name for c in self.SEGMENTS_COLUMNS} - {
            "raw_filename",
            "segment_onset",
            "segment_offset",
        }
        required_columns = {c.name for c in self.SEGMENTS_COLUMNS if c.required} - {
            "raw_filename",
            "segment_onset",
            "segment_offset",
        }
        assert union.issubset(
            all_columns
        ), "left_columns and right_columns have unexpected values"
        assert required_columns.issubset(
            union
        ), "left_columns and right_columns have missing values"

        annotations = self.annotations[
            self.annotations["set"].isin([left_set, right_set])
        ]
        annotations = annotations[annotations["error"].isnull()]

        intersection = AnnotationManager.intersection(
            annotations, sets=[left_set, right_set]
        )
        left_annotations = intersection[intersection["set"] == left_set]
        right_annotations = intersection[intersection["set"] == right_set]

        left_annotations = (
            left_annotations.reset_index(drop=True)
            .rename_axis("interval")
            .reset_index()
        )
        right_annotations = (
            right_annotations.reset_index(drop=True)
            .rename_axis("interval")
            .reset_index()
        )

        input_annotations = [
            {
                "left_annotations": left_annotations[
                    left_annotations["recording_filename"] == recording
                ],
                "right_annotations": right_annotations[
                    right_annotations["recording_filename"] == recording
                ],
            }
            for recording in left_annotations["recording_filename"].unique()
        ]

        pool = mp.Pool(processes=threads if threads > 0 else mp.cpu_count())
        annotations = pool.map(
            partial(
                self.merge_annotations, left_columns, right_columns, columns, output_set
            ),
            input_annotations,
        )
        annotations = pd.concat(annotations)
        annotations.drop(
            columns=list(
                set(annotations.columns) - {c.name for c in self.INDEX_COLUMNS}
            ),
            inplace=True,
        )
        annotations.fillna({"raw_filename": "NA"}, inplace=True)

        self.read()
        self.annotations = pd.concat([self.annotations, annotations], sort=False)
        self.write()

    def get_segments(self, annotations: pd.DataFrame) -> pd.DataFrame:
        """get all segments associated to the annotations referenced in ``annotations``.

        :param annotations: dataframe of annotations, according to :ref:`format-annotations`
        :type annotations: pd.DataFrame
        :return: dataframe of all the segments merged (as specified in :ref:`format-annotations-segments`), merged with ``annotations``. 
        :rtype: pd.DataFrame
        """
        annotations = annotations.dropna(subset=["annotation_filename"])
        annotations.drop(columns=["raw_filename"], inplace=True)

        segments = []
        for index, _annotations in annotations.groupby(["set", "annotation_filename"]):
            s, annotation_filename = index
            df = pd.read_csv(
                os.path.join(
                    self.project.path,
                    "annotations",
                    s,
                    "converted",
                    annotation_filename,
                )
            )

            for annotation in _annotations.to_dict(orient="records"):
                segs = df.copy()
                segs = AnnotationManager.clip_segments(
                    segs, annotation["range_onset"], annotation["range_offset"]
                )

                if not len(segs):
                    continue

                for c in annotation.keys():
                    segs[c] = annotation[c]

                segments.append(segs)

        return (
            pd.concat(segments)
            if segments
            else pd.DataFrame(
                columns=set(
                    [c.name for c in AnnotationManager.SEGMENTS_COLUMNS if c.required]
                    + list(annotations.columns)
                )
            )
        )

    def get_collapsed_segments(self, annotations: pd.DataFrame) -> pd.DataFrame:
        """get all segments associated to the annotations referenced in ``annotations``,
        and collapses into one virtual timeline.

        :param annotations: dataframe of annotations, according to :ref:`format-annotations`
        :type annotations: pd.DataFrame
        :return: dataframe of all the segments merged (as specified in :ref:`format-annotations-segments`), merged with ``annotations``
        :rtype: pd.DataFrame
        """
        annotations["duration"] = (
            annotations["range_offset"] - annotations["range_onset"]
        ).astype(float)

        annotations = annotations.sort_values(
            ["recording_filename", "range_onset", "range_offset", "set"]
        )
        annotations["position"] = annotations.groupby("set")["duration"].transform(
            pd.Series.cumsum
        )
        annotations["position"] = (
            annotations.groupby("set")["position"].shift(1).fillna(0)
        )

        segments = self.get_segments(annotations)

        segments["segment_onset"] += segments["position"] - segments["range_onset"]
        segments["segment_offset"] += segments["position"] - segments["range_onset"]

        return segments

    def get_within_time_range(
        self, annotations: pd.DataFrame, start_time: str, end_time: str, errors="raise"
    ):
        """Clip all input annotations within a given HH:MM clock-time range.
        Those that do not intersect the input time range at all are filtered out.

        :param annotations: DataFrame of input annotations to filter.
        The only columns that are required are: ``recording_filename``, ``range_onset``, and ``range_offset``.
        :type annotations: pd.DataFrame
        :param start: onset HH:MM clocktime
        :type start: str
        :param end: offset HH:MM clocktime
        :type end: str
        :param errors: how to deal with invalid start_time values for the recordings. Takes the same values as ``pandas.to_datetime``.
        :type errors: str
        :return: a DataFrame of annotations;
        For each row, ``range_onset`` and ``range_offset`` are clipped within the desired clock-time range.
        The clock-time corresponding to the onset and offset of each annotation
        is stored in two newly created columns named ``range_onset_time`` and ``range_offset_time``.
        If the input annotation exceeds 24 hours, one row per matching interval is returned.
        :rtype: pd.DataFrame
        """

        def get_ms_since_midight(dt):
            return (dt - dt.replace(hour=0, minute=0, second=0)).total_seconds() * 1000

        try:
            start_dt = datetime.datetime.strptime(start_time, "%H:%M")
        except:
            raise ValueError(
                f"invalid value for start_time ('{start_time}'); should have HH:MM format instead"
            )

        try:
            end_dt = datetime.datetime.strptime(end_time, "%H:%M")
        except:
            raise ValueError(
                f"invalid value for end_time ('{end_time}'); should have HH:MM format instead"
            )

        assert end_dt > start_dt, "end_time must follow start_time"

        start_ts = get_ms_since_midight(start_dt)
        end_ts = get_ms_since_midight(end_dt)

        annotations = annotations.merge(
            self.project.recordings[["recording_filename", "start_time"]], how="left"
        )
        annotations["start_time"] = pd.to_datetime(
            annotations["start_time"], format="%H:%M", errors=errors
        )

        # remove values with NaT start_time
        annotations.dropna(subset=["start_time"], inplace=True)

        # clock-time of the beginning and end of the annotation
        annotations["range_onset_time"] = annotations["start_time"] + pd.to_timedelta(
            annotations["range_onset"], unit="ms"
        )
        annotations["range_onset_ts"] = (
            annotations["range_onset_time"].apply(get_ms_since_midight).astype(int)
        )

        annotations["range_offset_ts"] = (
            annotations["range_onset_ts"]
            + annotations["range_offset"]
            - annotations["range_onset"]
        ).astype(int)

        matches = []
        for annotation in annotations.to_dict(orient="records"):
            onsets = np.arange(start_ts, annotation["range_offset_ts"], 86400 * 1000)
            offsets = onsets + (end_ts - start_ts)

            xs = (Segment(onset, offset) for onset, offset in zip(onsets, offsets))
            ys = iter(
                (Segment(annotation["range_onset_ts"], annotation["range_offset_ts"]),)
            )

            intersection = intersect_ranges(xs, ys)
            for segment in intersection:
                ann = annotation.copy()
                ann["range_onset"] += segment.start - ann["range_onset_ts"]
                ann["range_offset"] += segment.stop - ann["range_offset_ts"]

                ann["range_onset_time"] = str(
                    datetime.timedelta(milliseconds=segment.start % (86400 * 1000))
                )[:-3].zfill(len("00:00"))
                ann["range_offset_time"] = str(
                    datetime.timedelta(milliseconds=segment.stop % (86400 * 1000))
                )[:-3].zfill(len("00:00"))

                if ann["range_onset"] >= ann["range_offset"]:
                    continue

                matches.append(ann)

        if len(matches):
            return pd.DataFrame(matches).drop(
                columns=["range_onset_ts", "range_offset_ts"]
            )
        else:
            columns = set(annotations.columns) - {"range_onset_ts", "range_offset_ts"}
            return pd.DataFrame(columns=columns)

    def get_segments_timestamps(
        self,
        segments: pd.DataFrame,
        ignore_date: bool = False,
        onset: str = "segment_onset",
        offset: str = "segment_offset",
    ) -> pd.DataFrame:
        """Calculate the onset and offset clock-time of each segment

        :param segments: DataFrame of segments (as returned by :meth:`~ChildProject.annotations.AnnotationManager.get_segments`).
        :type segments: pd.DataFrame
        :param ignore_date: leave date information and use time data only, defaults to False
        :type ignore_date: bool, optional
        :param onset: column storing the onset timestamp in milliseconds, defaults to "segment_onset"
        :type onset: str, optional
        :param offset: column storing the offset timestamp in milliseconds, defaults to "segment_offset"
        :type offset: str, optional
        :return: Returns the input dataframe with two new columns ``onset_time`` and ``offset_time``.
        ``onset_time`` is a datetime object corresponding to the onset of the segment.
        ``offset_time`` is a datetime object corresponding to the offset of the segment.
        In case either ``start_time`` or ``date_iso`` is not specified for the corresponding recording,
        both values will be set to NaT.
        :rtype: pd.DataFrame
        """
        columns_to_merge = ["start_time"]
        if not ignore_date:
            columns_to_merge.append("date_iso")

        columns_to_drop = set(segments.columns) & set(columns_to_merge)

        if len(columns_to_drop):
            segments.drop(columns=columns_to_drop, inplace=True)

        segments = segments.merge(
            self.project.recordings[
                ["recording_filename"] + columns_to_merge
            ].set_index("recording_filename"),
            how="left",
            right_index=True,
            left_on="recording_filename",
        )

        if ignore_date:
            segments["start_time"] = pd.to_datetime(
                segments["start_time"], format="%H:%M", errors="coerce",
            )
        else:
            segments["start_time"] = pd.to_datetime(
                segments[["date_iso", "start_time"]].apply(
                    lambda row: "{} {}".format(
                        str(row["date_iso"]), str(row["start_time"])
                    ),
                    axis=1,
                ),
                format="%Y-%m-%d %H:%M",
                errors="coerce",
            )

        segments["onset_time"] = segments["start_time"] + pd.to_timedelta(
            segments[onset], unit="ms", errors="coerce"
        )
        segments["offset_time"] = segments["start_time"] + pd.to_timedelta(
            segments[offset], unit="ms", errors="coerce"
        )

        return segments.drop(columns=columns_to_merge)

    @staticmethod
    def intersection(annotations: pd.DataFrame, sets: list = None) -> pd.DataFrame:
        """Compute the intersection of all annotations for all sets and recordings,
        based on their ``recording_filename``, ``range_onset`` and ``range_offset``
        attributes. (Only these columns are required, but more can be passed and they
        will be preserved).

        :param annotations: dataframe of annotations, according to :ref:`format-annotations`
        :type annotations: pd.DataFrame
        :return: dataframe of annotations, according to :ref:`format-annotations`
        :rtype: pd.DataFrame
        """
        stack = []
        recordings = list(annotations["recording_filename"].unique())

        if sets is None:
            sets = list(annotations["set"].unique())
        else:
            annotations = annotations[annotations["set"].isin(sets)]

        for recording in recordings:
            _annotations = annotations[annotations["recording_filename"] == recording]
            _annotations = _annotations.sort_values(["range_onset", "range_offset"])

            segments = []
            for s in sets:
                ann = _annotations[_annotations["set"] == s]
                segments.append(
                    (
                        Segment(onset, offset)
                        for (onset, offset) in ann[
                            ["range_onset", "range_offset"]
                        ].values.tolist()
                    )
                )

            segments = reduce(intersect_ranges, segments)

            result = []
            for segment in segments:
                ann = _annotations.copy()
                ann["range_onset"].clip(
                    lower=segment.start, upper=segment.stop, inplace=True
                )
                ann["range_offset"].clip(
                    lower=segment.start, upper=segment.stop, inplace=True
                )
                ann = ann[(ann["range_offset"] - ann["range_onset"]) > 0]
                result.append(ann)

            if not len(result):
                continue

            _annotations = pd.concat(result)
            stack.append(_annotations)

        return pd.concat(stack) if len(stack) else pd.DataFrame()

    def set_from_path(self, path: str) -> str:
        annotations_path = os.path.join(self.project.path, 'annotations')

        if not path_is_parent(annotations_path, path):
            return None

        annotation_set = os.path.relpath(path, annotations_path)

        basename = os.path.basename(annotation_set)
        if basename == 'raw' or basename == 'converted':
            annotation_set = os.path.dirname(annotation_set)

        return annotation_set

    @staticmethod
    def clip_segments(segments: pd.DataFrame, start: int, stop: int) -> pd.DataFrame:
        """Clip all segments onsets and offsets within ``start`` and ``stop``.
        Segments outside of the range [``start``,``stop``] will be removed.

        :param segments: Dataframe of the segments to clip
        :type segments: pd.DataFrame
        :param start: range start (in milliseconds)
        :type start: int
        :param stop: range end (in milliseconds)
        :type stop: int
        :return: Dataframe of the clipped segments
        :rtype: pd.DataFrame
        """
        start = int(start)
        stop = int(stop)

        segments["segment_onset"].clip(lower=start, upper=stop, inplace=True)
        segments["segment_offset"].clip(lower=start, upper=stop, inplace=True)

        segments = segments[segments["segment_offset"] > segments["segment_onset"]]

        return segments
