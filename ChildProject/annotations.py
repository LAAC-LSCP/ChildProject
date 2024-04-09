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
import logging

from . import __version__
from .pipelines.derivations import DERIVATIONS
from .projects import ChildProject
from .converters import *
from .tables import IndexTable, IndexColumn, assert_dataframe, assert_columns_presence
from .utils import Segment, intersect_ranges, path_is_parent, TimeInterval, series_to_datetime, find_lines_involved_in_overlap


# Create a logger for the module (file)
logger_annotations = logging.getLogger(__name__)
# messages are propagated to the higher level logger (ChildProject), used in cmdline.py
logger_annotations.propagate = True

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
            description="source file to target. this field is dedicated to rttm and ALICE annotations that may combine annotations from several recordings into one same text file.",
            required=False,
        ),
        IndexColumn(
            name="annotation_filename",
            description="output formatted annotation location, relative to `annotations/<set>/converted` (automatic column, don't specify)",
            filename=True,
            required=False,
            generated=True,
        ),
        IndexColumn(
            name="imported_at",
            description="importation date (automatic column, don't specify)",
            datetime={"%Y-%m-%d %H:%M:%S"},
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
        IndexColumn(
            name="merged_from",
            description="sets used to generate this annotation by merging (comma separated)",
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
            description="vocal maturity defined as: C (canonical), N (non-canonical), Y (crying) L (laughing), J (junk), U (uncertain)",
            choices=["C", "N", "Y", "L", "J", "U","NA"],
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
            name="msc_type",
            description="morphosyntactical complexity of the utterances defined as: 0 (0 meaningful word), 1 (1 meaningful word), 2 (2 meaningful words), S (simple utterance), C (complex utterance), U (uncertain)",
            choices=["0", "1", "2", "S", "C", "U"],
        ),
        IndexColumn(
            name="gra_type",
            description="grammaticality of the utterances defined as: G (grammatical), J (ungrammatical), U (uncertain)",
            choices=["G", "J", "U"],
        ),
        IndexColumn(
            name="addressee",
            description="T if target-child-directed, C if other-child-directed, A if adult-directed, O if addressed to other, P if addressed to a pet, U if uncertain or other. Multiple values should be sorted and separated by commas",
            choices=["T", "C", "A", "O", "P", "U", "NA"],
        ),
        IndexColumn(
            name="transcription", description="orthographic transcription of the speech"
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
            name="lena_speaker",
            description="LENA speaker type",
            choices=[
                "TVF",
                "FAN",
                "OLN",
                "SIL",
                "NOF",
                "CXF",
                "OLF",
                "CHF",
                "MAF",
                "TVN",
                "NON",
                "CXN",
                "CHN",
                "MAN",
                "FAF",
            ],
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

        index_path = os.path.join(self.project.path, "metadata","annotations.csv")
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
                
        #check the index for overlaps, produces errors as the same set should not have overlaps in annotations
        ovl_ranges = find_lines_involved_in_overlap(self.annotations, labels=['recording_filename', 'set'])
        if ovl_ranges[ovl_ranges == True].shape[0] > 0:
            ovl_ranges = self.annotations[ovl_ranges][['set','annotation_filename']].values.tolist()
            errors.extend(
                [
                    f"overlaps in the annotation index for the following [set, annotation_filename] list: {ovl_ranges}"
                ]
            )
            
        #check the index for bad range_onset range_offset
        ranges_invalid = self.annotations[(self.annotations['range_offset'] <= self.annotations['range_onset']) | (self.annotations['range_onset'] < 0)]
        if ranges_invalid.shape[0] > 0:
            errors.extend(
                [f"annotation index does not verify range_offset > range_onset >= 0 for set <{line['set']}>, annotation filename <{line['annotation_filename']}>"
                  for line in ranges_invalid.to_dict(orient="records")]        
            )
            
        #if duration is in recordings.csv, check index for annotation segments overflowing recording duration
        if self.project.recordings is not None and 'duration' in self.project.recordings.columns:
            df = self.annotations.merge(self.project.recordings, how='left', on='recording_filename')
            ranges_invalid = df[(df['range_offset'] > df['duration'])]
            if ranges_invalid.shape[0] > 0:
                errors.extend(
                    [f"annotation index has an offset higher than recorded duration of the audio <{line['set']}>, annotation filename <{line['annotation_filename']}>"
                      for line in ranges_invalid.to_dict(orient="records")]        
                )
        
        warnings += self._check_for_outdated_merged_sets()

        return errors, warnings

    def validate_annotation(self, annotation: dict) -> Tuple[List[str], List[str]]:
        logger_annotations.info("Validating %s from %s...", annotation["annotation_filename"], annotation["set"])
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
        if annotations is None:
            annotations = self.annotations
        else:
            assert_dataframe("annotations", annotations)

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
        self.annotations.loc[:,["time_seek", "range_onset", "range_offset"]].fillna(
            0, inplace=True
        )
        self.annotations[
            ["time_seek", "range_onset", "range_offset"]
        ] = self.annotations[["time_seek", "range_onset", "range_offset"]].astype(np.int64)
        self.annotations = self.annotations.sort_values(['imported_at','set', 'annotation_filename'])
        self.annotations.to_csv(
            os.path.join(self.project.path, "metadata/annotations.csv"), index=False
        )
        
    def _check_for_outdated_merged_sets(self, sets: set = None):
        """Checks the annotations dataframe for sets that were used in merged sets and modified afterwards.
        This method produces warnings and suggestions to update the considered merged sets.
        
        :param sets: names of the original sets (sets used to merge) to consider.
        :type sets: set
        :return: List of warnings to give regarding the presence of outdated merged sets
        :rtype: List[str]
        """
        warnings = []
        
        #make a copy of annotation index, keep only the last modification date for each set (will not detect specific cases like partial merge)
        df = self.annotations.copy().sort_values(['set','imported_at']).groupby(['set']).last()
        
        #build a dictionary capturing the last modification date for each set.
        last_modif = {} 
        for i, row in df.iterrows():
            last_modif[i] = row['imported_at']
        
        #iterate through sets that were built from a merge and compare their last modification date to the one of their original set.
        if 'merged_from' in df.columns:
            merged_sets = df.dropna(subset=['merged_from'])[['merged_from', 'imported_at']]      
            for i, row in merged_sets.iterrows():
                for j in row['merged_from'].split(','):
                    #if a list of sets was given and the set is not in that list, skip it
                    if (sets is not None and j in sets) or sets is None:
                        if row['imported_at'] < last_modif[j]:
                            warnings.append("set {} is outdated because the {} set it is merged from was modified. Consider updating or rerunning the creation of the {} set.".format(i,j,i))
                        
        return warnings
    def _import_annotation(
        self, import_function: Callable[[str], pd.DataFrame],
        params: dict,
        annotation: dict,
        overwrite_existing: bool = False,
    ):
        """import and convert ``annotation``. This function should not be called outside of this class.

        :param import_function: If callable, ``import_function`` will be called to convert the input annotation into a dataframe. Otherwise, the conversion will be performed by a built-in function.
        :type import_function: Callable[[str], pd.DataFrame]
        :param params: Optional parameters. With ```new_tiers```, the corresponding EAF tiers will be imported
        :type params: dict
        :param annotation: input annotation dictionary (attributes defined according to :ref:`ChildProject.annotations.AnnotationManager.SEGMENTS_COLUMNS`)
        :type annotation: dict
        :param overwrite_existing: choose if lines with the same set and annotation_filename should be overwritten
        :type overwrite_existing: bool
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
        
        #check if the annotation file already exists in dataset (same filename and same set)
        if self.annotations[(self.annotations['set'] == annotation['set']) &
                            (self.annotations['annotation_filename'] == annotation_filename)].shape[0] > 0:
            if overwrite_existing:
                logger_annotations.warning("Annotation file %s will be overwritten", output_filename)
               
            else:
                error_filename = output_filename.replace('\\','/')
                annotation["error"] = f"annotation file {error_filename} already exists, to reimport it, use the overwrite_existing flag"
                logger_annotations.error("Error: %s", annotation['error'])
                annotation["imported_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                return annotation
        
        #find if there are annotation indexes in the same set that overlap the new annotation
        #as it is not possible to annotate multiple times the same audio stretch in the same set
        ovl_annots = self.annotations[(self.annotations['set'] == annotation['set']) &
                            (self.annotations['annotation_filename'] != annotation_filename) & #this condition avoid matching a line that should be overwritten (so has the same annotation_filename), it is dependent on the previous block!!!
                            (self.annotations['recording_filename'] == annotation['recording_filename']) &
                            (self.annotations['range_onset'] < annotation['range_offset']) &
                            (self.annotations['range_offset'] > annotation['range_onset']) 
                            ]
        if ovl_annots.shape[0] > 0:
            array_tup = list(ovl_annots[['set','recording_filename','range_onset', 'range_offset']].itertuples(index=False, name=None))
            annotation["error"] = f"importation for set <{annotation['set']}> recording <{annotation['recording_filename']}> from {annotation['range_onset']} to {annotation['range_offset']} cannot continue because it overlaps with these existing annotation lines: {array_tup}"
            logger_annotations.error("Error: %s", annotation['error'])
            #(f"Error: {annotation['error']}")
            annotation["imported_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return annotation
            
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
                df = converter.convert(path, filter, **params)
            else:
                raise ValueError(
                    "file format '{}' unknown for '{}'".format(annotation_format, path)
                )
        except:
            annotation["error"] = traceback.format_exc()
            logger_annotations.error("An error occurred while processing '%s'", path, exc_info=True)

        if df is None or not isinstance(df, pd.DataFrame):
            annotation["imported_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return annotation

        if not df.shape[1]:
            df = pd.DataFrame(columns=[c.name for c in self.SEGMENTS_COLUMNS])

        df["raw_filename"] = annotation["raw_filename"]

        df["segment_onset"] += np.int64(annotation["time_seek"])
        df["segment_offset"] += np.int64(annotation["time_seek"])
        df["segment_onset"] = df["segment_onset"].astype(np.int64)
        df["segment_offset"] = df["segment_offset"].astype(np.int64)

        annotation["time_seek"] = np.int64(annotation["time_seek"])
        annotation["range_onset"] = np.int64(annotation["range_onset"])
        annotation["range_offset"] = np.int64(annotation["range_offset"])

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
        new_tiers: list = None,
        overwrite_existing: bool = False,
    ) -> pd.DataFrame:
        """Import and convert annotations.

        :param input: dataframe of all annotations to import, as described in :ref:`format-input-annotations`.
        :type input: pd.DataFrame
        :param threads: If > 1, conversions will be run on ``threads`` threads, defaults to -1
        :type threads: int, optional
        :param import_function: If specified, the custom ``import_function`` function will be used to convert all ``input`` annotations, defaults to None
        :type import_function: Callable[[str], pd.DataFrame], optional
        :param new_tiers: List of EAF tiers names. If specified, the corresponding EAF tiers will be imported.
        :type new_tiers: list[str], optional
        :param overwrite_existing: choose if lines with the same set and annotation_filename should be overwritten
        :type overwrite_existing: bool, optional
        :return: dataframe of imported annotations, as in :ref:`format-annotations`.
        :rtype: pd.DataFrame
        """
        input_processed= input.copy().reset_index()
        
        required_columns = {
            c.name
            for c in AnnotationManager.INDEX_COLUMNS
            if c.required and not c.generated
        }

        assert_dataframe("input", input_processed)
        assert_columns_presence("input", input_processed, required_columns)
        
        input_processed["range_onset"] = input_processed["range_onset"].astype(np.int64)
        input_processed["range_offset"] = input_processed["range_offset"].astype(np.int64)
        
        assert (input_processed["range_offset"] > input_processed["range_onset"]).all(), "range_offset must be greater than range_onset"
        assert (input_processed["range_onset"] >= 0).all(), "range_onset must be greater or equal to 0"
        if "duration" in self.project.recordings.columns:
            assert (input_processed["range_offset"] <= input_processed.merge(self.project.recordings,
                                                                            how='left',
                                                                            on='recording_filename',
                                                                            validate='m:1'
                                                                            ).reset_index()["duration"]
            ).all(), "range_offset must be smaller than the duration of the recording"

        missing_recordings = input_processed[
            ~input_processed["recording_filename"].isin(
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
      
        builtin = input_processed[input_processed["format"].isin(converters.keys())]
        if not builtin["format"].map(lambda f: converters[f].THREAD_SAFE).all():
            logger_annotations.warning("warning: some of the converters do not support multithread importation; running on 1 thread")
            threads = 1
            
        #if the input to import has overlaps in it, raise an error immediately, nothing will be imported
        ovl_ranges = find_lines_involved_in_overlap(input_processed, labels=['recording_filename','set'])
        if ovl_ranges[ovl_ranges == True].shape[0] > 0:
            ovl_ranges = ovl_ranges[ovl_ranges].index.values.tolist()
            raise ValueError(f"the ranges given to import have overlaps on indexes : {ovl_ranges}")

        if threads == 1:
            imported = input_processed.apply(
                partial(self._import_annotation, import_function,
                                                {"new_tiers": new_tiers},
                                                overwrite_existing=overwrite_existing
                        ), axis=1
            ).to_dict(orient="records")
        else:
            with mp.Pool(processes=threads if threads > 0 else mp.cpu_count()) as pool:
                imported = pool.map(
                    partial(self._import_annotation, import_function,
                                                    {"new_tiers": new_tiers},
                                                    overwrite_existing=overwrite_existing
                    ),
                    input_processed.to_dict(orient="records"),
                )

        imported = pd.DataFrame(imported)
        imported.drop(
            list(set(imported.columns) - {c.name for c in self.INDEX_COLUMNS}),
            axis=1,
            inplace=True,
        )

        if 'error' in imported.columns:
            errors = imported[~imported["error"].isnull()]
            imported = imported[imported["error"].isnull()] 
            #when errors occur, separate them in a different csv in extra
            if errors.shape[0] > 0:
                output = os.path.join(self.project.path, "extra","errors_import_{}.csv".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
                errors.to_csv(output, index=False)
                logger_annotations.info("Errors summary exported to %s", output)              
        else:
            errors = None

        self.read()
        self.annotations = pd.concat([self.annotations, imported], sort=False)
        #at this point, 2 lines with same set and annotation_filename can happen if specified overwrite,
        # dropping duplicates remove the first importation and keeps the more recent one
        self.annotations = self.annotations.sort_values('imported_at').drop_duplicates(subset=["set","recording_filename","range_onset","range_offset"], keep='last')
        self.write()
        
        sets = set(input_processed['set'].unique())
        outdated_sets = self._check_for_outdated_merged_sets(sets= sets)
        for warning in outdated_sets:
            logger_annotations.warning("warning: %s", warning)          

        return (imported, errors)

    def _derive_annotation(
            self,
            annotation: dict,
            import_function: Callable,
            output_set: str,
            overwrite_existing: bool = False,
    ):
        """import and convert ``annotation``. This function should not be called outside of this class.

        :param annotation: input annotation dictionary (attributes defined according to :ref:`ChildProject.annotations.AnnotationManager.SEGMENTS_COLUMNS`)
        :type annotation: dict
        :param import_function: derivation function to apply to the original set. This can be a python function or a string name of a stored method in .pipelines.derivations.DERIVATIONS .
        :type import_function: Callable[[str], pd.DataFrame]
        :param output_set: name of the new set of derived annotations
        :type output_set: str
        :param overwrite_existing: use for lines with the same set and annotation_filename to be rederived and overwritten
        :type overwrite_existing: bool
        :return: output annotation dictionary (attributes defined according to :ref:`ChildProject.annotations.AnnotationManager.SEGMENTS_COLUMNS`)
        :rtype: dict
        """
        annotation_result = annotation.copy()
        annotation_result['set'] = output_set
        annotation_result['format'] = "NA"
        annotation_result['merged_from'] = annotation['set']

        source_recording = os.path.splitext(annotation["recording_filename"])[0]
        annotation_filename = "{}_{}_{}.csv".format(
            source_recording, annotation["range_onset"], annotation["range_offset"]
        )
        output_filename = os.path.join(
            "annotations", output_set, "converted", annotation_filename
        )

        # check if the annotation file already exists in dataset (same filename and same set)
        if self.annotations[(self.annotations['set'] == output_set) &
                            (self.annotations['annotation_filename'] == annotation_filename)].shape[0] > 0:
            if overwrite_existing:
                logger_annotations.warning("Derived file %s will be overwritten", output_filename)

            else:
                logger_annotations.warning("File %s already exists. To overwrite, specify parameter ''overwrite_existing'' to True", output_filename)
                return annotation_result

        # find if there are annotation indexes in the same set that overlap the new annotation
        # as it is not possible to annotate multiple times the same audio stretch in the same set
        ovl_annots = self.annotations[(self.annotations['set'] == output_set) &
                                      (self.annotations[
                                           'annotation_filename'] != annotation_filename) &  # this condition avoid matching a line that should be overwritten (so has the same annotation_filename), it is dependent on the previous block!!!
                                      (self.annotations['recording_filename'] == annotation['recording_filename']) &
                                      (self.annotations['range_onset'] < annotation['range_offset']) &
                                      (self.annotations['range_offset'] > annotation['range_onset'])
                                      ]
        if ovl_annots.shape[0] > 0:
            array_tup = list(
                ovl_annots[['set', 'recording_filename', 'range_onset', 'range_offset']].itertuples(index=False,
                                                                                                    name=None))
            annotation_result[
                "error"] = f"derivation for set <{output_set}> recording <{annotation['recording_filename']}> from {annotation['range_onset']} to {annotation['range_offset']} cannot continue because it overlaps with these existing annotation lines: {array_tup} . You could try removing entirely the set and trying again."
            logger_annotations.error("Error: %s", annotation['error'])
            # (f"Error: {annotation['error']}")
            annotation_result["imported_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return annotation_result

        path = os.path.join(
            self.project.path,
            "annotations",
            annotation["set"],
            "converted", #EXPAND
            annotation["annotation_filename"],
        )

        #TODO CHECK FOR DTYPES, enforcing dtypes of converted annotation files will become standard in a future update
        df_input = pd.read_csv(path)
        df = None

        def bad_derivation(annotation_dict, msg_err, error, path_file):
            annotation_dict["error"] = traceback.format_exc()
            logger_annotations.error("An error occurred while processing '%s': %s", path_file, msg_err, exc_info=True)
            annotation_dict["imported_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return annotation_dict

        # get all the metadata into a single dictionary for the derivation function
        rec_dict = self.project.recordings[
            self.project.recordings['recording_filename'] == annotation['recording_filename']].iloc[0].to_dict()
        chi_dict = self.project.children[
            self.project.children['child_id'] == rec_dict['child_id']].iloc[0].to_dict()
        metadata = {**chi_dict, **rec_dict, **annotation}

        # apply the derivation to the annotation dataframe
        # if the derivation raises an exception stop the processing there and return the line
        try:
            df = import_function(self.project, metadata, df_input)
        except Exception as e:
            return bad_derivation(annotation_result, e, traceback.format_exc(), path)

        # if the derivation function did not return a dataframe, stop there and return the line
        if df is None or not isinstance(df, pd.DataFrame):
            msg = f"<{import_function}> did not return a pandas DataFrame"
            return bad_derivation(annotation_result, msg, msg, path)

        # if the derivation does not contain the required columns of annotations
        if not {c.name for c in self.SEGMENTS_COLUMNS if c.required}.issubset(df.columns):
            required = {c.name for c in self.SEGMENTS_COLUMNS if c.required}
            msg = f"DataFrame result of <{import_function}> function does not contain the required {required}"
            return bad_derivation(annotation_result, msg, msg, path)

        if not df.shape[1]:
            df = pd.DataFrame(columns=[c.name for c in self.SEGMENTS_COLUMNS])

        df["raw_filename"] = annotation["raw_filename"]

        df["segment_onset"] += np.int64(annotation["time_seek"])
        df["segment_offset"] += np.int64(annotation["time_seek"])
        df["segment_onset"] = df["segment_onset"].astype(np.int64)
        df["segment_offset"] = df["segment_offset"].astype(np.int64)

        annotation_result["time_seek"] = np.int64(annotation["time_seek"])
        annotation_result["range_onset"] = np.int64(annotation["range_onset"])
        annotation_result["range_offset"] = np.int64(annotation["range_offset"])

        df = AnnotationManager.clip_segments(
            df, annotation_result["range_onset"], annotation_result["range_offset"]
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

        annotation_result["annotation_filename"] = annotation_filename
        annotation_result["imported_at"] = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        annotation_result["package_version"] = __version__

        return annotation_result

    def derive_annotations(self,
                           input_set: str,
                           output_set: str,
                           derivation_function: Union[str, Callable],
                           threads: int = -1,
                           overwrite_existing: bool = False,
                           ) -> (pd.DataFrame, pd.DataFrame):
        """Derive annotations. From an existing set of annotations, create a new set that derive its result from
        the original set

        :param input_set: name of the set of annotations to be derived
        :rtype: str
        :param output_set: name of the new set of derived annotations
        :rtype: str
        :param derivation_function: name of the derivation type to be performed
        :rtype: Union[str, Callable]
        :param threads: If > 1, conversions will be run on ``threads`` threads, defaults to -1
        :type threads: int, optional
        :param overwrite_existing: choice if lines with the same set and annotation_filename should be overwritten
        :type overwrite_existing: bool, optional
        :return: tuple of dataframe of derived annotations, as in :ref:`format-annotations` and dataframe of errors
        :rtype: tuple (pd.DataFrame, pd.DataFrame)
        """
        input_processed = self.annotations[self.annotations['set'] == input_set].copy()
        assert not input_processed.empty, "Input set {0} does not exist,\
         existing sets are in the 'set' column of annotations.csv".format(input_set)

        assert input_set != output_set, "Input set {0} should be different than output\
         set {1}".format(input_set, output_set)

        # check the existence of the derivation function and that it is callable or predefined
        if not callable(derivation_function):
            if derivation_function in DERIVATIONS.keys():
                derivation_function = DERIVATIONS[derivation_function]
            else:
                raise ValueError(
                    "derivation value '{}' unknown, use one of {}".format(derivation_function, DERIVATIONS.keys())
                )

        if threads == 1:
            # apply the derivation function to each annotation file that needs to be derived (sequential)
            imported = input_processed.apply(
                partial(self._derive_annotation,
                        import_function=derivation_function,
                        output_set=output_set,
                        overwrite_existing=overwrite_existing
                        ), axis=1
            ).to_dict(orient="records")
        else:
            # apply the derivation function to each annotation file that needs to be derived (threaded)
            with mp.Pool(processes=threads if threads > 0 else mp.cpu_count()) as pool:
                imported = pool.map(
                    partial(self._derive_annotation,
                            import_function=derivation_function,
                            output_set=output_set,
                            overwrite_existing=overwrite_existing
                            ),
                    input_processed.to_dict(orient="records"),
                )

        imported = pd.DataFrame(imported)
        # drop additional columns that are not supposed to be kept in annotations.csv
        imported.drop(
            list(set(imported.columns) - {c.name for c in self.INDEX_COLUMNS}),
            axis=1,
            inplace=True,
        )

        # if at least 1 error occurred
        if 'error' in imported.columns:
            # separate importations that resulted in error from successful ones
            errors = imported[~imported["error"].isnull()]
            imported = imported[imported["error"].isnull()]
            # when errors occur, separate them in a different csv in extra
            if errors.shape[0] > 0:
                output = os.path.join(self.project.path, "extra",
                                      "errors_derive_{}.csv".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
                errors.to_csv(output, index=False)
                logger_annotations.info("Errors summary exported to %s", output)
        else:
            errors = None

        # here we add the new lines of imported annotations to the annotations.csv file
        self.read()
        self.annotations = pd.concat([self.annotations, imported], sort=False)
        # at this point, 2 lines with same set and annotation_filename can happen if specified overwrite,
        # dropping duplicates remove the first importation and keeps the more recent one
        self.annotations = self.annotations.sort_values('imported_at').drop_duplicates(
            subset=["set", "recording_filename", "range_onset", "range_offset"], keep='last')
        self.write()

        sets = set(input_processed['set'].unique())

        # this block should be executed everytime we change annotations.csv
        # it checks that sets that are derived or merged from others are not outdated
        # (meaning their importation is more recent than the merge/derivation)
        outdated_sets = self._check_for_outdated_merged_sets(sets=sets)
        for warning in outdated_sets:
            logger_annotations.warning("warning: %s", warning)

        return imported, errors

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
            logger_annotations.info("could not delete '%s', as it does not exist (yet?)", path)
            pass

        self.annotations = self.annotations[self.annotations["set"] != annotation_set]
        self.write()
      
        outdated_sets = self._check_for_outdated_merged_sets(sets= {annotation_set})
        for warning in outdated_sets:
            logger_annotations.warning("warning: %s", warning)

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
            if os.path.exists(os.path.join(new_path, 'raw')):
                raise Exception("raw folder '{}' already exists, aborting".format(os.path.join(new_path, 'raw')))
            if os.path.exists(os.path.join(new_path, 'converted')):
                raise Exception("converted folder '{}' already exists, aborting".format(os.path.join(new_path, 'converted')))
                
        if (self.annotations[self.annotations["set"] == new_set].shape[0] > 0): 
            raise Exception("'{}' set already exists in the index".format(new_set))

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
                
        #find the merged from lines that should be updated and update them
        if 'merged_from' in self.annotations.columns:
            merged_from = self.annotations['merged_from'].astype(str).str.split(',')
            matches = [False if not isinstance(s, list) else annotation_set in s for s in merged_from.values.tolist()]
            
            def update_mf(old_list, old, new):
                res = set(old_list)
                res.discard(old)
                res.add(new)
                return ','.join(res)
            
            self.annotations.loc[matches, 'merged_from'] = merged_from[matches].apply(partial(update_mf, old=annotation_set,new=new_set))
        self.write()

    def merge_annotations(
        self, left_columns, right_columns, columns, output_set, input, skip_existing: bool = False
    ):
        """From 2 DataFrames listing the annotation indexes to merge together (those indexes should come from
        the intersection of the left_set and right_set indexes), the listing of the columns
        to merge and name of the output_set, creates the resulting csv files containing the converted merged
        segments and returns the new indexes to add to annotations.csv.

        :param left_columns: list of the columns to include from the left set
        :type left_columns: list[str]
        :param right_columns: list of the columns to include from the right set
        :type right_columns: list[str]
        :param columns: additional columns to add to the segments, key is the column name
        :type columns: dict
        :param output_set: name of the set to save the new merged files into
        :type output_set: str
        :param input: annotation indexes to use for the merge, contains keys 'left_annotations' and 'right_annotations' to separate indexes from left and right set
        :type input: dict
        :param input:
        :type input: bool
        :return: annotation indexes created by the merge, should be added to annotations.csv
        :rtype: pandas.DataFrame
        """
        #get the left and right annotation dataframes
        left_annotations = input["left_annotations"]
        right_annotations = input["right_annotations"]
        
        #start the new annotations from a copy of the left set
        annotations = left_annotations.copy()
        #store the annotation filenames used to keep a reference to those existing files
        annotations['left_annotation_filename'] = annotations["annotation_filename"]
        annotations['right_annotation_filename'] = right_annotations['annotation_filename']
        #populate the raw_filename column with the raw filenames of the sets used to merge, separated by a comma
        annotations['raw_filename'] = left_annotations['raw_filename'] + ',' + right_annotations['raw_filename']
        #package version is the version used by the merge, not the one used in the merged sets
        annotations["package_version"] = __version__
        
        #format of a merged set is undefined
        annotations["format"] = "NA"
        #compute the names of the new annotation filenames that will be created
        annotations["annotation_filename"] = annotations.apply(
            lambda annotation: "{}_{}_{}.csv".format(
                os.path.splitext(annotation["recording_filename"])[0],
                annotation["range_onset"],
                annotation["range_offset"],
            ),
            axis=1,
        )
        #store in 'merged_from' the names of the sets it was merged from
        annotations['merged_from'] = ','.join(np.concatenate([left_annotations['set'].unique() , right_annotations['set'].unique()]))
        #the timestamps will be recomputed from the start of the file, so time_seek is always 0 on a merged set
        annotations['time_seek'] = 0
        
        #if skip existing, only keep the line where the resulting converted file does not already exist (even as a broken symlink)
        if skip_existing:
            annotations = annotations[~annotations['annotation_filename'].map(lambda x : os.path.lexists(os.path.join(self.project.path,"annotations",output_set, "converted", x)))]
            left_annotations = left_annotations[left_annotations['annotation_filename'].isin(annotations['left_annotation_filename'].to_list())]
            right_annotations = right_annotations[right_annotations['annotation_filename'].isin(annotations['right_annotation_filename'].to_list())]
        
        for key in columns:
            annotations[key] = columns[key]

        annotations["set"] = output_set

        #check the presence of the converted files in the left_set
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

        #check the presence of the converted files in the right_set
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

        #get the actual annotation segments
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

        #merge left and right annotations segments
        output_segments = left_segments[list(lc)].merge(
            right_segments[list(rc)],
            how="outer",
            left_on=merge_columns,
            right_on=merge_columns,
        )

        output_segments["segment_onset"] = (
            output_segments["segment_onset"].fillna(0).astype(np.int64)
        )
        output_segments["segment_offset"] = (
            output_segments["segment_offset"].fillna(0).astype(np.int64)
        )
        
        output_segments["raw_filename"] = (
            output_segments["raw_filename_x"].fillna("")
            + "," 
            + output_segments["raw_filename_y"].fillna("")
        )
        output_segments["raw_filename"] = output_segments["raw_filename"].str.strip(',')
        output_segments.drop(
            columns=["raw_filename_x", "raw_filename_y", "time_seek"], inplace=True
        )

        #drop unused columns, get the currect datetime and store it in imported_at
        annotations.drop(columns=['right_annotation_filename', 'left_annotation_filename'], inplace=True)
        annotations["imported_at"] = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        

        output_segments.fillna("NA", inplace=True)

        #create the new converted files from the merged annotation segments
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
        full_set_merge: bool = True,
        skip_existing: bool = False,
        columns: dict = {},
        recording_filter: str = None,
        threads=-1,
    ):
        """Merge columns from ``left_set`` and ``right_set`` annotations, 
        for all matching segments, into a new set of annotations named
        ``output_set`` that will be saved in the dataset. ``output_set``
        must not already exist if full_set_merge is True. 

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
        :param full_set_merge: The merge is meant to create the entired merged set. Therefore, the set should not already exist. defaults to True
        :type full_set_merge: bool
        :param skip_existing: The merge will skip already existing lines in the merged set. So both the annotation index and resulting converted csv will not change for those lines
        :type skip_existing: bool
        :param columns: Additional columns to add to the resulting converted annotations.
        :type columns: dict
        :param recording_filter: set of recording_filenames to merge.
        :type recording_filter: set[str]
        :param threads: number of threads
        :type threads: int
        :return: [description]
        :rtype: [type]
        """
        existing_sets = self.annotations['set'].unique()
        if full_set_merge: assert output_set not in existing_sets, "output_set <{}> already exists, remove the existing set or another name.".format(output_set)
        assert left_set in existing_sets, "left_set <{}> was not found, check the spelling.".format(left_set)
        assert right_set in existing_sets, "right_set <{}> was not found, check the spelling.".format(right_set)
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
        
        if recording_filter:
            annotations = annotations[annotations['recording_filename'].isin(recording_filter)]

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
                self.merge_annotations, left_columns, right_columns, columns, output_set, skip_existing=skip_existing
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
        # if annotations.csv can have duplicate entries with same converted filename and is normal, check this https://stackoverflow.com/a/45927402 and change the code
        self.annotations = pd.concat([self.annotations, annotations], sort=False).drop_duplicates(subset=['set','recording_filename','annotation_filename'], keep='last')
        self.write()

    def get_segments(self, annotations: pd.DataFrame) -> pd.DataFrame:
        """get all segments associated to the annotations referenced in ``annotations``.

        :param annotations: dataframe of annotations, according to :ref:`format-annotations`
        :type annotations: pd.DataFrame
        :return: dataframe of all the segments merged (as specified in :ref:`format-annotations-segments`), merged with ``annotations``. 
        :rtype: pd.DataFrame
        """
        assert_dataframe("annotations", annotations)
        assert_columns_presence(
            "annotations",
            annotations,
            {
                "annotation_filename",
                "raw_filename",
                "set",
                "range_onset",
                "range_offset",
            },
        )

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
                columns=list(set(
                    [c.name for c in AnnotationManager.SEGMENTS_COLUMNS if c.required]
                    + list(annotations.columns))
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
        assert_dataframe("annotations", annotations)
        assert_columns_presence(
            "annotations",
            annotations,
            {"range_onset", "range_offset", "recording_filename", "set",},
        )

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

    def get_within_ranges(
        self,
        ranges: pd.DataFrame,
        sets: Union[Set, List] = None,
        missing_data: str = "ignore",
    ):
        """Retrieve and clip annotations that cover specific portions of recordings (``ranges``).
        
        The desired ranges are defined by an input dataframe with three columns: ``recording_filename``, ``range_onset``, and ``range_offset``.
        The function returns a dataframe of annotations under the same format as the index of annotations (:ref:`format-annotations`).
       
        This output get can then be provided to :meth:`~ChildProject.annotations.AnnotationManager.get_segments`
        in order to retrieve segments of annotations that match the desired range.

        For instance, the code belows will prints all the segments of annotations
        corresponding to the first hour of each recording:

        >>> from ChildProject.projects import ChildProject
        >>> from ChildProject.annotations import AnnotationManager
        >>> project = ChildProject('.')
        >>> am = AnnotationManager(project)
        >>> am.read()
        >>> ranges = project.recordings
        >>> ranges['range_onset'] = 0
        >>> ranges['range_offset'] = 60*60*1000
        >>> matches = am.get_within_ranges(ranges)
        >>> am.get_segments(matches)

        :param ranges: pandas dataframe with one row per range to be considered and three columns: ``recording_filename``, ``range_onset``, ``range_offset``.
        :type ranges: pd.DataFrame
        :param sets: optional list of annotation sets to retrieve. If None, all annotations from all sets will be retrieved.
        :type sets: Union[Set, List]
        :param missing_data: how to handle missing annotations ("ignore", "warn" or "raise")
        :type missing_data: str, defaults to ignore
        :rtype: pd.DataFrame
        """

        assert_dataframe("ranges", ranges)
        assert_columns_presence(
            "ranges", ranges, {"recording_filename", "range_onset", "range_offset"}
        )

        if sets is None:
            sets = set(self.annotations["set"].tolist())
        else:
            sets = set(sets)
            missing = sets - set(self.annotations["set"].tolist())
            if len(missing):
                raise ValueError(
                    "the following sets are missing from the annotations index: {}".format(
                        ",".join(missing)
                    )
                )

        annotations = self.annotations[self.annotations["set"].isin(sets)]

        stack = []
        recordings = list(ranges["recording_filename"].unique())

        for recording in recordings:
            _ranges = ranges[ranges["recording_filename"] == recording].sort_values(
                ["range_onset", "range_offset"]
            )
            _annotations = annotations[
                annotations["recording_filename"] == recording
            ].sort_values(["range_onset", "range_offset"])

            for s in sets:
                ann = _annotations[_annotations["set"] == s]

                selected_segments = (
                    Segment(onset, offset)
                    for (onset, offset) in _ranges[
                        ["range_onset", "range_offset"]
                    ].values.tolist()
                )

                set_segments = (
                    Segment(onset, offset)
                    for (onset, offset) in ann[
                        ["range_onset", "range_offset"]
                    ].values.tolist()
                )

                intersection = intersect_ranges(selected_segments, set_segments)

                segments = []
                for segment in intersection:
                    segment_ann = ann.copy()
                    segment_ann["range_onset"].clip(
                        lower=segment.start, upper=segment.stop, inplace=True
                    )
                    segment_ann["range_offset"].clip(
                        lower=segment.start, upper=segment.stop, inplace=True
                    )
                    segment_ann = segment_ann[
                        (segment_ann["range_offset"] - segment_ann["range_onset"]) > 0
                    ]
                    segments.append(segment_ann.copy())

                stack += segments

                if missing_data == "ignore":
                    continue

                duration = 0
                if segments:
                    segments = pd.concat(segments)
                    duration = (
                        segments["range_offset"] - segments["range_onset"]
                    ).sum()

                selected_duration = (
                    _ranges["range_offset"] - _ranges["range_onset"]
                ).sum()

                if duration >= selected_duration:
                    continue

                error_message = (
                    f"""annotations from set '{s}' do not cover the whole selected range """
                    f"""for recording '{recording}', """
                    f"""{duration/1000:.3f}s covered instead of {selected_duration/1000:.3f}s"""
                )

                if missing_data == "warn":
                    logger_annotations.warning("warning: %s", error_message)
                else:
                    raise Exception(error_message)

        return pd.concat(stack) if len(stack) else pd.DataFrame()

    def get_within_time_range(self,
        annotations: pd.DataFrame,
        interval : TimeInterval = None,
        start_time: str = None,
        end_time: str = None,
    ):
        """Clip all input annotations within a given HH:MM:SS clock-time range.
        Those that do not intersect the input time range at all are filtered out.

        :param annotations: DataFrame of input annotations to filter. The only columns that are required are: ``recording_filename``, ``range_onset``, and ``range_offset``.
        :type annotations: pd.DataFrame
        :param interval: Interval of hours to consider, contains the start hour and end hour
        :type interval: TimeInterval
        :param start_time: start_time to use in a HH:MM format, only used if interval is None, replaces the first value of interval
        :type start_time: str
        :param end_time: end_time to use in a HH:MM format, only used if interval is None, replaces the second value of interval
        :type end_time: str
        :return: a DataFrame of annotations; \
        For each row, ``range_onset`` and ``range_offset`` are clipped within the desired clock-time range. \
        The clock-time corresponding to the onset and offset of each annotation \
        is stored in two newly created columns named ``range_onset_time`` and ``range_offset_time``. \
        If the input annotation exceeds 24 hours, one row per matching interval is returned. \
        :rtype: pd.DataFrame
        """
        assert interval is not None or (start_time and end_time), "you must pass an interval or a start_time and end_time"
        
        if interval is None:
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
            interval = TimeInterval(start_dt,end_dt)

        assert_dataframe("annotations", annotations)
        assert_columns_presence(
            "annotations",
            annotations,
            {"recording_filename", "range_onset", "range_offset"},
        )

        def get_ms_since_midight(dt):
            return (dt - dt.replace(hour=0, minute=0, second=0)).total_seconds() * 1000

        #assert end_dt > start_dt, "end_time must follow start_time"
        # no reason to keep this condition, 23:00 to 03:00 is completely acceptable

        if not isinstance(interval, TimeInterval): raise ValueError("interval must be a TimeInterval object")
        start_ts = get_ms_since_midight(interval.start)
        end_ts = get_ms_since_midight(interval.stop)

        annotations = annotations.merge(
            self.project.recordings[["recording_filename", "start_time"]], how="left"
        )
        
        annotations['start_time'] = series_to_datetime(
                annotations['start_time'], ChildProject.RECORDINGS_COLUMNS, 'start_time'
        )

        # remove values with NaT start_time
        annotations.dropna(subset=["start_time"], inplace=True)

        # clock-time of the beginning and end of the annotation
        annotations["range_onset_time"] = annotations["start_time"] + pd.to_timedelta(
            annotations["range_onset"], unit="ms"
        )
        annotations["range_onset_ts"] = (
            annotations["range_onset_time"].apply(get_ms_since_midight).astype('int64')
        )

        annotations["range_offset_ts"] = (
            annotations["range_onset_ts"]
            + annotations["range_offset"]
            - annotations["range_onset"]
        ).astype(np.int64)

        matches = []
        for annotation in annotations.to_dict(orient="records"):
            #onsets = np.arange(start_ts, annotation["range_offset_ts"], 86400 * 1000)
            #offsets = onsets + (end_ts - start_ts)
            
            onsets = np.arange(start_ts, annotation["range_offset_ts"], 86400 * 1000)
            offsets = np.arange(end_ts, annotation["range_offset_ts"], 86400 * 1000)
            #treat edge cases when the offset is after the end of annotation, onset before start etc
            if len(onsets) > 0 and onsets[0] < annotation["range_onset_ts"] :
                if len(offsets) > 0 and offsets[0] < annotation["range_onset_ts"]: onsets = onsets[1:]
                else : onsets[0] = annotation["range_onset_ts"]
            if len(offsets) > 0 and offsets[0] < annotation["range_onset_ts"] : offsets = offsets[1:]
            if len(onsets) > 0 and len(offsets) > 0 and onsets[0] > offsets[0] : onsets = np.append(annotation["range_onset_ts"], onsets)
            if (len(onsets) > 0 and len(offsets) > 0 and onsets[-1] > offsets[-1]) or len(onsets) > len(offsets) : offsets = np.append(offsets,annotation["range_offset_ts"])

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
            columns = list(set(annotations.columns) - {"range_onset_ts", "range_offset_ts"})
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
        :return: Returns the input dataframe with two new columns ``onset_time`` and ``offset_time``. \
        ``onset_time`` is a datetime object corresponding to the onset of the segment. \
        ``offset_time`` is a datetime object corresponding to the offset of the segment. \
        In case either ``start_time`` or ``date_iso`` is not specified for the corresponding recording, \
        both values will be set to NaT.
        :rtype: pd.DataFrame
        """

        assert_dataframe("segments", segments)
        assert_columns_presence(
            "segments", segments, {"recording_filename", onset, offset}
        )

        columns_to_merge = ["start_time"]
        if not ignore_date:
            columns_to_merge.append("date_iso")

        columns_to_drop = list(set(segments.columns) & set(columns_to_merge))

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
            segments['start_time'] = series_to_datetime(
                segments['start_time'], ChildProject.RECORDINGS_COLUMNS, 'start_time'
            )
        else:
            segments['start_time'] = series_to_datetime(
                segments['start_time'], ChildProject.RECORDINGS_COLUMNS, 'start_time', date_series = segments['date_iso'], date_index_list = ChildProject.RECORDINGS_COLUMNS, date_column_name = 'date_iso'
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
        assert_dataframe("annotations", annotations)
        assert_columns_presence(
            "annotations",
            annotations,
            {"recording_filename", "set", "range_onset", "range_offset"},
        )

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
        annotations_path = os.path.join(self.project.path, "annotations")

        if not path_is_parent(annotations_path, path):
            return None

        annotation_set = os.path.relpath(path, annotations_path)

        basename = os.path.basename(annotation_set)
        if basename == "raw" or basename == "converted":
            annotation_set = os.path.dirname(annotation_set)

        return annotation_set.replace("\\","/")

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
        assert_dataframe("segments", segments)
        assert_columns_presence(
            "segments", segments, {"segment_onset", "segment_offset"}
        )

        start = int(start)
        stop = int(stop)

        segments["segment_onset"].clip(lower=start, upper=stop, inplace=True)
        segments["segment_offset"].clip(lower=start, upper=stop, inplace=True)

        segments = segments[segments["segment_offset"] > segments["segment_onset"]]

        return segments
