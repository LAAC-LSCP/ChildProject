import datetime
from functools import partial
import glob
import numpy as np
import os
import pandas as pd
import re
import subprocess

from .tables import (
    IndexTable,
    IndexColumn,
    is_boolean,
    assert_dataframe,
    assert_columns_presence,
)
from .utils import get_audio_duration, path_is_parent

RAW_RECORDINGS = os.path.normpath("recordings/raw")
CONVERTED_RECORDINGS = os.path.normpath("recordings/converted")
STANDARD_SAMPLE_RATE = 16000
STANDARD_PROFILE = 'standard' # profile that is expected to contain the standardized audios (16kHz). The existence and sampling rates of this profile are checked when <validating this profile> or <validating without profile and the raw recordings are not 16kHz>.

METADATA_FOLDER = 'metadata'
CHILDREN_CSV = 'children.csv'
RECORDINGS_CSV = 'recordings.csv'

PROJECT_FOLDERS = ["recordings", "annotations", "metadata", "doc", "scripts"]


class ChildProject:
    """ChildProject instance
    This class is a representation of a ChildProject dataset

    Constructor parameters:

    :param path: path to the root of the dataset.
    :type path: str
    :param enforce_dtypes: enforce dtypes on children/recordings dataframes, defaults to False
    :type enforce_dtypes: bool, optional
    :param ignore_discarded: ignore entries such that discard=1, defaults to True
    :type ignore_discarded: bool, optional
    
    Attributes:
    :param path: path to the root of the dataset.
    :type path: str
    :param recordings: pandas dataframe representation of this dataset metadata/recordings.csv 
    :type recordings: class:`pd.DataFrame`
    :param children: pandas dataframe representation of this dataset metadata/children.csv 
    :type children: class:`pd.DataFrame`
    """

    REQUIRED_DIRECTORIES = ["recordings", "extra"]

    CHILDREN_COLUMNS = [
        IndexColumn(
            name="experiment",
            description="one word to capture the unique ID of the data collection effort; for instance Tsimane_2018, solis-intervention-pre",
            required=True,
        ),
        IndexColumn(
            name="child_id",
            description="unique child ID -- unique within the experiment (Id could be repeated across experiments to refer to different children)",
            unique=True,
            required=True,
            dtype="str",
        ),
        IndexColumn(
            name="child_dob",
            description="child's date of birth",
            required=True,
            datetime={"%Y-%m-%d"},
        ),
        IndexColumn(
            name="location_id",
            description="Unique location ID -- only specify here if children never change locations in this culture; otherwise, specify in the recordings metadata",
        ),
        IndexColumn(
            name="child_sex",
            description="f= female, m=male",
            choices=["m", "M", "f", "F"],
        ),
        IndexColumn(
            name="language",
            description='language the child is exposed to if child is monolingual; small caps, indicate dialect by name or location if available; eg "france french"; "paris french"',
        ),
        IndexColumn(
            name="languages",
            description='list languages child is exposed to separating them with ; and indicating the percentage if one is available; eg: "french 35%; english 65%"',
        ),
        IndexColumn(name="mat_ed", description="maternal years of education"),
        IndexColumn(name="fat_ed", description="paternal years of education"),
        IndexColumn(
            name="car_ed",
            description="years of education of main caregiver (if not mother or father)",
        ),
        IndexColumn(
            name="monoling",
            description="whether the child is monolingual (Y) or not (N)",
            choices=["Y", "N"],
        ),
        IndexColumn(
            name="monoling_criterion",
            description='how monoling was decided; eg "we asked families which languages they spoke in the home"',
        ),
        IndexColumn(
            name="normative",
            description="whether the child is normative (Y) or not (N)",
            choices=["Y", "N"],
        ),
        IndexColumn(
            name="normative_criterion",
            description='how normative was decided; eg "unless the caregivers volunteered information whereby the child had a problem, we consider them normative by default"',
        ),
        IndexColumn(name="mother_id", description="unique ID of the mother"),
        IndexColumn(name="father_id", description="unique ID of the father"),
        IndexColumn(
            name="order_of_birth",
            description="child order of birth",
            regex=r"(\d+(\.\d+)?)",
            required=False,
        ),
        IndexColumn(
            name="n_of_siblings",
            description="amount of siblings",
            regex=r"(\d+(\.\d+)?)",
            required=False,
        ),
        IndexColumn(
            name="household_size",
            description="number of people living in the household (adults+children)",
            regex=r"(\d+(\.\d+)?)",
            required=False,
        ),
        IndexColumn(
            name="dob_criterion",
            description="determines whether the date of birth is known exactly or extrapolated e.g. from the age. Dates of birth are assumed to be known exactly if this column is NA or unspecified.",
            choices=["extrapolated", "exact"],
            required=False,
        ),
        IndexColumn(
            name="dob_accuracy",
            description="date of birth accuracy",
            choices=["day", "week", "month", "year", "other"],
        ),
        IndexColumn(
            name="discard",
            description="set to 1 if item should be discarded in analyses",
            choices=["0", "1"],
            required=False,
        ),
    ]

    RECORDINGS_COLUMNS = [
        IndexColumn(
            name="experiment",
            description="one word to capture the unique ID of the data collection effort; for instance Tsimane_2018, solis-intervention-pre",
            required=True,
        ),
        IndexColumn(
            name="child_id",
            description="unique child ID -- unique within the experiment (Id could be repeated across experiments to refer to different children)",
            required=True,
            dtype="str",
        ),
        IndexColumn(
            name="date_iso",
            description="date in which recording was started in ISO (eg 2020-09-17)",
            required=True,
            datetime={"%Y-%m-%d"},
        ),
        IndexColumn(
            name="start_time",
            description="local time in which recording was started in format 24-hour (H)H:MM:SS or (H)H:MM; if minutes or seconds are unknown, use 00. ‘NA’ if unknown, this will raise a Warning when validating as some analysis that rely on times will not consider this recordings.",
            required=True,
            datetime={"%H:%M","%H:%M:%S"},
        ),
        IndexColumn(
            name="recording_device_type",
            description="lena, usb, olympus, babylogger (lowercase), izyrec",
            required=True,
            choices=["lena", "usb", "olympus", "babylogger", "izyrec", "unknown"],
        ),
        IndexColumn(
            name="recording_filename",
            description="the path to the file from the root of “recordings”). It MUST be unique (two recordings cannot point towards the same file).",
            required=True,
            filename=True,
            unique=True,
            dtype="str",
        ),
        IndexColumn(
            name="duration",
            description="duration of the audio, in milliseconds",
            regex=r"([0-9]+)",
        ),
        IndexColumn(
            name="session_id",
            description="identifier of the recording session.",
            dtype="str",
        ),
        IndexColumn(
            name="session_offset",
            description="offset (in milliseconds) of the recording with respect to other recordings that are part of the same session. Each recording session is identified by their `session_id`.",
            regex=r"[0-9]+",
        ),
        IndexColumn(
            name="recording_device_id", description="unique ID of the recording device"
        ),
        IndexColumn(
            name="experimenter",
            description="who collected the data (could be anonymized ID)",
        ),
        IndexColumn(
            name="location_id",
            description="unique location ID -- can be specified at the level of the child (if children do not change locations)",
        ),
        IndexColumn(name="its_filename", description="its_filename"),
        IndexColumn(name="upl_filename", description="upl_filename"),
        IndexColumn(name="trs_filename", description="trs_filename"),
        IndexColumn(name="lena_id", description=""),
        IndexColumn(
            name="lena_recording_num",
            description="value of the corresponding <Recording> num's attribute, for LENA recordings that have been split into contiguous parts",
            dtype="int",
        ),
        IndexColumn(
            name="might_feature_gaps",
            description="1 if the audio cannot be guaranteed to be a continuous block with no time jumps, 0 or NA or undefined otherwise.",
            function=is_boolean,
        ),
        IndexColumn(
            name="start_time_accuracy",
            description="Accuracy of start_time for this recording. If not specified, assumes second-accuray.",
            choices=["second", "minute", "hour", "reliable"],
        ),
        IndexColumn(
            name="noisy_setting",
            description="1 if the audio may be noisier than the childs usual day, 0 or undefined otherwise",
            function=is_boolean,
        ),
        IndexColumn(
            name="notes",
            description="free-style notes about individual recordings (avoid tabs and newlines)",
        ),
        IndexColumn(
            name="discard",
            description="set to 1 if item should be discarded in analyses",
            choices=["0", "1"],
            required=False,
        ),
    ]

    DOCUMENTATION_COLUMNS = [
        IndexColumn(
            name="variable",
            description="name of the variable",
            unique=True,
            required=True,
        ),
        IndexColumn(
            name="description", description="a definition of this field", required=True
        ),
        IndexColumn(name="values", description="a summary of authorized values"),
        IndexColumn(name="scope", description="which group of users has access to it"),
        IndexColumn(
            name="annotation_set",
            description="for annotations: which set(s) contain this variable",
        ),
    ]
    
    REC_COL_REF = {c.name: c for c in RECORDINGS_COLUMNS}

    def __init__(
        self, path: str, enforce_dtypes: bool = False, ignore_discarded: bool = True
    ):
        self.path = path
        self.experiment = None
        self.enforce_dtypes = enforce_dtypes
        self.ignore_discarded = ignore_discarded

        self.errors = []
        self.warnings = []
        self.children = None
        self.recordings = None

        self.children_metadata_origin = None
        self.recordings_metadata_origin = None

        self.converted_recordings_hashtable = {}

    def accumulate_metadata(
        self,
        table: str,
        df: pd.DataFrame,
        columns: list,
        merge_column: str,
        verbose=False,
    ) -> pd.DataFrame:
        md_path = os.path.join(self.path, METADATA_FOLDER, table)

        if not os.path.exists(md_path):
            return df

        md = pd.DataFrame(
            [
                {"path": f, "basename": os.path.basename(f)}
                for f in glob.glob(os.path.join(md_path, "**/*.csv"), recursive=True)
            ]
        )

        if not len(md):
            return df

        md.sort_values("basename", ascending=False, inplace=True)

        duplicates = md.groupby("basename").agg(
            paths=("path", list), count=("path", len),
        )
        duplicates = duplicates[duplicates["count"] >= 2].reset_index()

        if len(duplicates):
            raise Exception(
                "ambiguous filenames detected:\n{}".format(
                    "\n".join(
                        duplicates.apply(
                            lambda d: "{} found as {}".format(
                                ",".join(d["basename"]), d["paths"]
                            ),
                            axis=1,
                        ).tolist()
                    )
                )
            )

        for md in md["path"].tolist():
            if not os.path.exists(md):
                continue

            table = IndexTable(table, md, columns, enforce_dtypes=self.enforce_dtypes)
            dataframe = table.read()

            replaced_columns = (set(df.columns) & set(dataframe.columns)) - {
                merge_column
            }
            if verbose and len(replaced_columns):
                print(
                    "column(s) {} overwritten by {}".format(
                        ",".join(replaced_columns), md
                    )
                )

            df["line"] = df.index
            df = (
                df[list((set(df.columns) - set(dataframe.columns)) | {merge_column})]
                .merge(
                    dataframe, how="left", left_on=merge_column, right_on=merge_column
                )
                .set_index("line")
            )

        return df

    def read(self, verbose=False, accumulate=True):
        """Read the metadata from the project and stores it in recordings and children attributes
        
        :param verbose: read with additional output
        :type verbose: bool
        :param accumulate: add metadata from subfolders (usually confidential metadata)
        :type accumulate: bool
        """
        self.ct = IndexTable(
            "children",
            os.path.join(self.path, METADATA_FOLDER,CHILDREN_CSV),
            self.CHILDREN_COLUMNS,
            enforce_dtypes=self.enforce_dtypes,
        )
        self.rt = IndexTable(
            "recordings",
            os.path.join(self.path, METADATA_FOLDER,RECORDINGS_CSV),
            self.RECORDINGS_COLUMNS,
            enforce_dtypes=self.enforce_dtypes,
        )

        self.children = self.ct.read()
        self.recordings = self.rt.read()

        # accumulate additional metadata (optional)
        if accumulate:
            self.ct.df = self.accumulate_metadata(
                "children", self.children, self.CHILDREN_COLUMNS, "child_id", verbose
            )
            self.rt.df = self.accumulate_metadata(
                "recordings",
                self.recordings,
                self.RECORDINGS_COLUMNS,
                "recording_filename",
                verbose,
            )

        if self.ignore_discarded and "discard" in self.ct.df:
            self.ct.df['discard'] = self.ct.df["discard"].apply(np.nan_to_num).astype(int, errors='ignore')
            self.ct.df = self.ct.df[self.ct.df["discard"].astype(str) != "1"]

        if self.ignore_discarded and "discard" in self.rt.df:
            self.rt.df['discard'] = self.rt.df["discard"].apply(np.nan_to_num).astype(int, errors='ignore')
            self.rt.df = self.rt.df[self.rt.df["discard"].astype(str) != "1"]

        self.children = self.ct.df
        self.recordings = self.rt.df
        
        exp = self.children.iloc[0]['experiment']
        exp_values = set(self.children['experiment'].unique()).union(set(self.recordings['experiment'].unique()))
        if len(exp_values) > 1:
            raise ValueError(f"Column <experiment> must be unique across the dataset, in both children.csv and recordings.csv , {len(exp_values)} different values were found: {exp_values}")
        self.experiment = exp
        
    def write_recordings(self, keep_discarded: bool = True, keep_original_columns: bool = True):
        """
        Write self.recordings to the recordings csv file of the dataset.
        !! if `read()` was done with `accumulate` , you may write confidential information in recordings.csv !!
        
        :param keep_discarded: if True, the lines in the csv that are discarded by the dataset are kept when writing. defaults to True (when False, discarded lines disappear from the dataset)
        :type keep_discarded: bool, optional
        :param keep_original_columns: if True, deleting columns in the recordings dataframe will not result in them disappearing from the csv file (if false, only the current columns are kept)
        :type keep_original_columns: bool, optional
        :return: dataframe that was written to the csv file
        :rtype: pandas.DataFrame
        """
        if self.recordings is None:
            #logger to add (can not write recordings file as recordings is not initialized)
            return None
        #get the file as reference point
        current_csv = pd.read_csv(os.path.join(self.path, METADATA_FOLDER,RECORDINGS_CSV))
        
        if 'discard' in current_csv.columns and keep_discarded:
            # put the discard column into a usable form
            current_csv['discard'] = current_csv['discard'].apply(np.nan_to_num).astype(int, errors='ignore')
            # keep the discarded lines somewhere
            discarded_recs = current_csv[current_csv['discard'].astype(str) == "1"]
            
            recs_to_write = pd.concat([self.recordings,discarded_recs])
            recs_to_write = recs_to_write.astype(self.recordings.dtypes.to_dict())
        else:
            recs_to_write = self.recordings
            
        if keep_original_columns:
            columns = current_csv.columns
            for new in self.recordings.columns:
                if new not in columns : columns = columns.append(pd.Index([new]))
        else:
            columns = self.recordings.columns
            
        recs_to_write.sort_index().to_csv(os.path.join(self.path, METADATA_FOLDER, RECORDINGS_CSV),columns = columns,index=False)
        return recs_to_write

    def validate(self, ignore_recordings: bool = False, profile: str = None, accumulate: bool = True) -> tuple:
        """Validate a dataset, returning all errors and warnings.

        :param ignore_recordings: if True, no errors will be returned for missing recordings.
        :type ignore_recordings: bool, optional
        :param profile: profile of recordings to use
        :type profile: str, optional
        :param accumulate: use accumulated metadata (usually confidential metadata if present)
        :type accumualte: bool, optional
        :return: A tuple containing the list of errors, and the list of warnings.
        :rtype: a tuple of two lists
        """
        self.errors = []
        self.warnings = []

        directories = [d for d in os.listdir(self.path) if os.path.isdir(self.path)]

        for rd in self.REQUIRED_DIRECTORIES:
            if rd not in directories:
                self.errors.append("missing directory {}.".format(rd))

        # check tables
        self.read(verbose=True, accumulate=accumulate)

        errors, warnings = self.ct.validate()
        self.errors += errors
        self.warnings += warnings

        errors, warnings = self.rt.validate()
        self.errors += errors
        self.warnings += warnings

        if ignore_recordings:
            return self.errors, self.warnings

        from pydub.utils import mediainfo #mediainfo to get audio files info
        for index, row in self.recordings.iterrows():
            
            # make sure that recordings exist
            for column_name in self.recordings.columns:
                column_attr = next(
                    (c for c in self.RECORDINGS_COLUMNS if c.name == column_name), None
                )

                if column_attr is None:
                    continue

                if column_attr.filename and row[column_name] != "NA":
                    raw_filename = str(row[column_name])

                    try:
                        path = self.get_recording_path(raw_filename, profile)
                    except:
                        if profile:
                            profile_metadata = os.path.join(
                                self.path,
                                CONVERTED_RECORDINGS,
                                profile,
                                RECORDINGS_CSV,
                            )
                            self.errors.append(
                                f"failed to recover the path for recording '{raw_filename}' and profile '{profile}'. Does the profile exist? Does {profile_metadata} exist?"
                            )
                        continue

                    if os.path.exists(path):
                        if not profile:
                            info = mediainfo(path)
                            if 'sample_rate' not in info or int(info['sample_rate']) != STANDARD_SAMPLE_RATE:
                                try:
                                    std_path = self.get_recording_path(raw_filename, STANDARD_PROFILE)
                                    if os.path.exists(std_path):
                                        std_info = mediainfo(std_path)
                                        if 'sample_rate' not in std_info:
                                            self.warnings.append(
                                                f"Could not read the sample rate of converted version of recording '{raw_filename}' at '{std_path}'. {STANDARD_SAMPLE_RATE}Hz is expected for profile {STANDARD_PROFILE}")
                                        elif int(std_info['sample_rate']) != STANDARD_SAMPLE_RATE:
                                            self.warnings.append(f"converted version of recording '{raw_filename}' at '{std_path}' has unexpected sampling rate {std_info['sample_rate']}Hz when {STANDARD_SAMPLE_RATE}Hz is expected for profile {STANDARD_PROFILE}")
                                    else:
                                        if 'sample_rate' in info:
                                            self.warnings.append(
                                                f"recording '{raw_filename}' at '{path}' has a non standard sampling rate of {info['sample_rate']}Hz and no standard conversion in profile {STANDARD_PROFILE} was found. Does the standard profile exist? Does {profile_metadata} exist? you can create the standard profile with 'child-project process {self.path} {STANDARD_PROFILE} basic --format=wav --sampling={STANDARD_SAMPLE_RATE} --codec=pcm_s16le --skip-existing'")
                                        else:
                                            self.warnings.append(
                                                f"Could not read the sample rate of recording '{raw_filename}' at '{path}' and no standard conversion in profile {STANDARD_PROFILE} was found. Does the standard profile exist? Does {profile_metadata} exist? you can create the standard profile with 'child-project process {self.path} {STANDARD_PROFILE} basic --format=wav --sampling={STANDARD_SAMPLE_RATE} --codec=pcm_s16le --skip-existing'")
                                except:
                                    profile_metadata = os.path.join(self.path,CONVERTED_RECORDINGS,STANDARD_PROFILE,RECORDINGS_CSV,)
                                    if 'sample_rate' in info:
                                        self.warnings.append(f"recording '{raw_filename}' at '{path}' has a non standard sampling rate of {info['sample_rate']}Hz and no standard conversion in profile {STANDARD_PROFILE} was found. Does the standard profile exist? Does {profile_metadata} exist? you can create the standard profile with 'child-project process {self.path} {STANDARD_PROFILE} basic --format=wav --sampling={STANDARD_SAMPLE_RATE} --codec=pcm_s16le --skip-existing'")
                                    else:
                                        self.warnings.append(f"Could not read the sample rate of recording '{raw_filename}' at '{path}' and no standard conversion in profile {STANDARD_PROFILE} was found. Does the standard profile exist? Does {profile_metadata} exist? you can create the standard profile with 'child-project process {self.path} {STANDARD_PROFILE} basic --format=wav --sampling={STANDARD_SAMPLE_RATE} --codec=pcm_s16le --skip-existing'")
                        elif profile == STANDARD_PROFILE:
                            info = mediainfo(path)
                            if 'sample_rate' in info and int(info['sample_rate']) != STANDARD_SAMPLE_RATE:
                                self.warnings.append(f"recording '{raw_filename}' at '{path}' has unexpected sampling rate {info['sample_rate']}Hz when {STANDARD_SAMPLE_RATE}Hz is expected for profile {STANDARD_PROFILE}")
                    
                    elif os.path.islink(path):
                        message = self.warnings.append(f"the data content of recording '{raw_filename}' at path '{path}' is absent. See 'broken symlinks'") #The path is valid but there's no content. See broken symlinks (try 'Datalad get $filename')
                    else:    
                        message = f"cannot find recording '{raw_filename}' at '{path}'"
                        if column_attr.required:
                            self.errors.append(message)
                        else:
                            self.warnings.append(message)

            # child id refers to an existing child in the children table
            if (
                str(row["child_id"])
                not in self.children["child_id"].astype(str).tolist()
            ):
                self.errors.append(
                    "child_id '{}' in recordings on line {} cannot be found in the children table.".format(
                        row["child_id"], index
                    )
                )

        # consistency between dates of birth and recording dates
        if "date_iso" in self.recordings.columns and "child_dob" in self.children.columns:
            ages = self.compute_ages(
                recordings=self.recordings,
                children=self.children.drop_duplicates(["child_id"], keep="first")
            )

            self.errors += [
                f"Age at recording is negative in recordings on line {index} ({age:.1f} months). Check date_iso for that recording and child_dob for the corresponding child."
                for index, age in ages[ages < 0].items()
            ]

        # detect un-indexed recordings and throw warnings
        files = [
            self.recordings[c.name].tolist()
            for c in self.RECORDINGS_COLUMNS
            if c.filename and c.name in self.recordings.columns
        ]

        indexed_files = [
            os.path.abspath(os.path.join(self.path, RAW_RECORDINGS, str(f)))
            for f in pd.core.common.flatten(files)
        ]

        recordings_files = glob.glob(
            os.path.join(os.path.normcase(self.path), RAW_RECORDINGS, "**/*.*"), recursive=True
        )

        for rf in recordings_files:
            if len(os.path.splitext(rf)) > 1 and os.path.splitext(rf)[1] in [
                ".csv",
                ".xls",
                ".xlsx",
            ]:
                continue

            ap = os.path.abspath(rf)
            if ap not in indexed_files:
                self.warnings.append("file '{}' not indexed.".format(rf))

        return self.errors, self.warnings

    def get_recording_path(self, recording_filename: str, profile: str = None) -> str:
        """return the path to a recording

        :param recording_filename: recording filename, as in the metadata
        :type recording_filename: str
        :param profile: name of the conversion profile, defaults to None
        :type profile: str, optional
        :return: path to the recording
        :rtype: str
        """

        if profile:
            converted_filename = self.get_converted_recording_filename(
                profile, recording_filename
            )

            if converted_filename is None:
                return None

            return os.path.join(
                os.path.normcase(self.path), CONVERTED_RECORDINGS, profile, os.path.normpath(converted_filename),
            )
        else:
            return os.path.join(os.path.normcase(self.path), RAW_RECORDINGS, os.path.normpath(recording_filename))

    def get_converted_recording_filename(
        self, profile: str, recording_filename: str
    ) -> str:
        """retrieve the converted filename of a recording under a given ``profile``,
        from its original filename.

        :param profile: recording profile
        :type profile: str
        :param recording_filename: original recording filename, as indexed in the metadata
        :type recording_filename: str
        :return: corresponding converted filename of the recording under this profile
        :rtype: str
        """

        key = (profile, recording_filename)

        if key in self.converted_recordings_hashtable:
            return self.converted_recordings_hashtable[key]

        converted_recordings = pd.read_csv(
            os.path.join(
                self.path, CONVERTED_RECORDINGS, profile, RECORDINGS_CSV
            )
        )
        converted_recordings.dropna(subset=["converted_filename"], inplace=True)

        self.converted_recordings_hashtable.update(
            {
                (profile, original): converted
                for original, converted in converted_recordings.loc[
                    :, ["original_filename", "converted_filename"]
                ].values
            }
        )

        if key in self.converted_recordings_hashtable:
            return self.converted_recordings_hashtable[key]
        else:
            self.converted_recordings_hashtable[key] = None
            return None

    def recording_from_path(self, path: str, profile: str = None) -> str:
        if profile:
            raise NotImplementedError(
                "cannot recover recording from the path to a converted media yet"
            )
            # media_path = os.path.join(self.path, CONVERTED_RECORDINGS, profile)
        else:
            media_path = os.path.join(self.path, RAW_RECORDINGS)

        if not path_is_parent(media_path, path):
            return None

        recording = os.path.relpath(path, media_path)

        return recording

    def get_recordings_from_list(
        self, recordings: list, profile: str = None
    ) -> pd.DataFrame:
        """Recover recordings metadata from a list of recordings or path to recordings.

        :param recordings: list of recording names or paths
        :type recordings: list
        :return: matching recordings
        :rtype: pd.DataFrame
        """
        _recordings = self.recordings.copy()
        _recordings = _recordings[
            (~_recordings["recording_filename"].isnull())
            & (_recordings["recording_filename"] != "NA")
        ]

        if recordings is not None:
            # if the user provided paths,
            # transform those paths into recording_filename values
            recordings_from_paths = [
                self.recording_from_path(recording, profile) for recording in recordings
            ]

            if None not in recordings_from_paths:
                recordings = recordings_from_paths

            _recordings = _recordings[
                _recordings["recording_filename"].isin(recordings)
            ]
            
            if _recordings.shape[0] < len(recordings):
                recs = pd.Series(recordings)
                missing_recs = recs[~recs.isin(self.recordings['recording_filename'])].tolist()
                #self.recordings[~self.recordings['recording_filename'].isin(recordings)]['recording_filename'].tolist()
                raise ValueError("recordings {} were not found in the dataset index. Check the names and make sure they exist in '{}'".format(missing_recs,os.path.join(METADATA_FOLDER,RECORDINGS_CSV)))
                

        return _recordings

    def compute_recordings_duration(self, profile: str = None) -> pd.DataFrame:
        """compute recordings duration

        :param profile: name of the profile of recordings to compute the duration from. If None, raw recordings are used. defaults to None
        :type profile: str, optional
        :return: dataframe of the recordings, with an additional/updated duration columns.
        :rtype: pd.DataFrame
        """
        recordings = self.recordings[["recording_filename"]]

        recordings = recordings.assign(
            duration=recordings["recording_filename"].map(
                lambda f: get_audio_duration(self.get_recording_path(f, profile))
            )
        )
        recordings["duration"].fillna(0, inplace=True)
        recordings["duration"] = (recordings["duration"] * 1000).astype(np.int64)

        return recordings

    def compute_ages(
        self,
        recordings: pd.DataFrame = None,
        children: pd.DataFrame = None,
        age_format: str = 'months',
    ) -> pd.Series:
        """Compute the age of the subject child for each recording (in months, as a float)
        and return it as a pandas Series object.

        Example:

        >>> from ChildProject.projects import ChildProject
        >>> project = ChildProject("examples/valid_raw_data")
        >>> project.read()
        >>> project.recordings["age"] = project.compute_ages()
        >>> project.recordings[["child_id", "date_iso", "age"]]
            child_id    date_iso       age
        line                                
        2            1  2020-04-20  3.613963
        3            1  2020-04-21  3.646817

        :param recordings: custom recordings DataFrame (see :ref:`format-metadata`), otherwise use all project recordings, defaults to None
        :type recordings: pd.DataFrame, optional
        :param children: custom children DataFrame (see :ref:`format-metadata`), otherwise use all project children data, defaults to None
        :type children: pd.DataFrame, optional
        :param age_format: format to use for the output date default is months, choose between ['months','days','weeks', 'years']
        :type age_format: str, optional
        """

        def date_is_valid(date: str, fmt: str):
            try:
                datetime.datetime.strptime(date, fmt)
            except:
                return False
            return True
        
        def date_fmt(dt,fmt='months'):
            if dt:
                if fmt == 'months':
                    return dt.days / (365.25 / 12)
                elif fmt == 'days':
                    return dt.days
                elif fmt == 'weeks':
                    return dt.days / 7
                elif fmt == 'years':
                    return dt.days / 365.25
                else:
                    raise ValueError('unknown format for age : {}'.format(fmt))
            else:
                return None

        if recordings is None:
            recordings = self.recordings.copy()

        if children is None:
            children = self.children.copy()

        assert_dataframe("recordings", recordings)
        assert_dataframe("children", children)

        assert_columns_presence("recordings", recordings, {"date_iso", "child_id"})
        assert_columns_presence("children", children, {"child_dob", "child_id"})

        index = recordings.index
        recordings = recordings.merge(
            children[["child_id", "child_dob"]],
            how="left",
            left_on="child_id",
            right_on="child_id",
        )
        recordings.index = index

        age = (
            recordings[["date_iso", "child_dob"]]
            .apply(
                lambda r: (
                    datetime.datetime.strptime(r["date_iso"], "%Y-%m-%d")
                    - datetime.datetime.strptime(r["child_dob"], "%Y-%m-%d")
                )
                if (
                    date_is_valid(r["child_dob"], "%Y-%m-%d")
                    and date_is_valid(r["date_iso"], "%Y-%m-%d")
                )
                else None,
                axis=1,
            )
            .apply(partial(date_fmt,fmt=age_format))
        )

        return age

    def read_documentation(self) -> pd.DataFrame:
        docs = ["children", "recordings", "annotations"]

        documentation = []

        for doc in docs:
            path = os.path.join(self.path, "docs", f"{doc}.csv")

            if not os.path.exists(path):
                continue

            table = IndexTable(f"{doc}-documentation", path, self.DOCUMENTATION_COLUMNS)
            table.read()
            documentation.append(table.df.assign(table=doc))

        documentation = pd.concat(documentation)
        return documentation
