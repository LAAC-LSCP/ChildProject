from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from ChildProject.converters import *
from functools import partial

import pandas as pd
import numpy as np
import datetime
import os
import pytest
import shutil
from pathlib import Path
import time


def standardize_dataframe(df, columns):
    df = df[list(columns)]
    return df.sort_index(axis=1).sort_values(list(columns)).reset_index(drop=True)


DATA = Path('tests', 'data')
TRUTH = Path('tests', 'truth')
PATH = Path('output', 'annotations')


@pytest.fixture(scope="function")
def project(request):
    if os.path.exists(PATH):
        # shutil.copytree(src="examples/valid_raw_data", dst="output/annotations")
        shutil.rmtree(PATH)
    shutil.copytree(src="examples/valid_raw_data", dst=PATH)

    project = ChildProject(PATH)

    yield project


@pytest.fixture(scope="function")
def am(request, project):
    am = AnnotationManager(project)
    # force longer durations to allow for imports
    project.recordings['duration'] = [100000000, 2000000]
    yield am


def test_csv():
    converted = CsvConverter().convert("tests/data/csv.csv").fillna("NA")
    truth = pd.read_csv("tests/truth/csv.csv").fillna("NA")

    pd.testing.assert_frame_equal(
        standardize_dataframe(converted, converted.columns),
        standardize_dataframe(truth, converted.columns),
        check_dtype=False,
    )


def test_vtc():
    converted = VtcConverter().convert("tests/data/vtc.rttm")
    truth = pd.read_csv("tests/truth/vtc.csv").fillna("NA")

    pd.testing.assert_frame_equal(
        standardize_dataframe(converted, converted.columns),
        standardize_dataframe(truth, converted.columns),
        check_dtype=False,
    )


def test_vcm():
    converted = VcmConverter().convert("tests/data/vcm.rttm")
    truth = pd.read_csv("tests/truth/vcm.csv").fillna("NA")

    pd.testing.assert_frame_equal(
        standardize_dataframe(converted, converted.columns),
        standardize_dataframe(truth, converted.columns),
        check_dtype=False,
    )


def test_alice():
    converted = AliceConverter().convert("tests/data/alice.txt")
    truth = pd.read_csv("tests/truth/alice.csv").fillna("NA")

    pd.testing.assert_frame_equal(
        standardize_dataframe(converted, converted.columns),
        standardize_dataframe(truth, converted.columns),
    )


def test_eaf():
    converted = EafConverter().convert("tests/data/eaf.eaf")
    truth = pd.read_csv("tests/truth/eaf.csv", dtype={"transcription": str}).fillna(
        "NA"
    )

    pd.testing.assert_frame_equal(
        standardize_dataframe(converted, converted.columns),
        standardize_dataframe(truth, converted.columns),
    )


def test_eaf_any_tier():
    converted = EafConverter().convert("tests/data/eaf_any_tier.eaf", new_tiers=['newtier', 'newtier2']) \
        .replace('', "NA") \
        .fillna("NA")
    truth = pd.read_csv("tests/truth/eaf_any_tier.csv", dtype={"transcription": str}).fillna(
        "NA"
    )

    pd.testing.assert_frame_equal(
        standardize_dataframe(converted, converted.columns),
        standardize_dataframe(truth, converted.columns),
    )


def test_textgrid():
    converted = TextGridConverter().convert("tests/data/textgrid.TextGrid")
    truth = pd.read_csv("tests/truth/textgrid.csv", dtype={"ling_type": str}).fillna(
        "NA"
    )

    pd.testing.assert_frame_equal(
        standardize_dataframe(converted, converted.columns),
        standardize_dataframe(truth, converted.columns),
    )


def test_cha():
    converted = ChatConverter.convert("tests/data/vandam.cha")
    truth = pd.read_csv("tests/truth/cha.csv").fillna("NA")

    pd.testing.assert_frame_equal(
        standardize_dataframe(converted, converted.columns),
        standardize_dataframe(truth, converted.columns),
    )


@pytest.mark.parametrize("its", ["example_lena_new", "example_lena_old"])
def test_its(its):
    converted = ItsConverter().convert(os.path.join("tests/data", its + ".its"))
    truth = pd.read_csv(
        os.path.join("tests/truth/its", "{}_ITS_Segments.csv".format(its)), dtype={'recordingInfo': 'object'}
    )  # .fillna('NA')
    check_its(converted, truth)


@pytest.mark.parametrize("nline,column,value,exception,error",
                         [(0, 'range_onset', -10, AssertionError, "range_onset must be greater or equal to 0"),
                          (3, 'range_offset', 1970000, AssertionError, "range_offset must be greater than range_onset"),
                          ])
def test_rejected_imports(project, nline, column, value, exception, error):
    am = AnnotationManager(project)

    input_annotations = pd.read_csv("examples/valid_raw_data/annotations/input.csv")

    input_annotations.iloc[nline, input_annotations.columns.get_loc(column)] = value

    print(input_annotations[['range_onset', 'range_offset']])

    with pytest.raises(exception, match=error):
        am.import_annotations(input_annotations)


def test_import(project, am):
    am.read()
    original_number = am.annotations.shape[0]
    input_annotations = pd.read_csv("examples/valid_raw_data/annotations/input.csv")
    am.import_annotations(input_annotations)


    assert (
            am.annotations.shape[0] == input_annotations.shape[0] + original_number
    ), "imported annotations length does not match input"

    assert all(
        [
            os.path.exists(
                os.path.join(
                    project.path,
                    "annotations",
                    a["set"],
                    "converted",
                    a["annotation_filename"],
                )
            )
            for a in am.annotations.to_dict(orient="records")
        ]
    ), "some annotations are missing"

    errors, warnings = am.validate()
    assert len(errors) == 0 and len(warnings) == 0, "malformed annotations detected"

    errors, warnings = am.read()
    assert len(errors) == 0 and len(warnings) == 0, "malformed annotation indexes detected"

    for dataset in ["eaf_basic", "textgrid", "eaf_solis"]:
        annotations = am.annotations[am.annotations["set"] == dataset]
        segments = am.get_segments(annotations)
        segments.drop(columns=set(annotations.columns) - {"raw_filename"}, inplace=True)
        truth = pd.read_csv("tests/truth/{}.csv".format(dataset))

        print(segments)
        print(truth)

        pd.testing.assert_frame_equal(
            standardize_dataframe(segments, set(truth.columns.tolist())),
            standardize_dataframe(truth, set(truth.columns.tolist())),
            check_exact=False,
            rtol=1e-3
        )


# test how the importation handles already existing files and overlaps in importation
@pytest.mark.parametrize("input_file,ow,rimported,rerrors,exception",
                         [("input_invalid.csv", False, None, None, ValueError),
                          ("input_reimport.csv", False, "imp_reimport_no_ow.csv", "err_reimport_no_ow.csv", None),
                          ("input_reimport.csv", True, "imp_reimport_ow.csv", None, None),
                          ("input_importoverlaps.csv", False, "imp_overlap.csv", "err_overlap.csv", None),
                          ("input_import_duration_overflow.csv", False, None, None, AssertionError),
                          ])
def test_multiple_imports(project, am, input_file, ow, rimported, rerrors, exception):
    am.read()
    original_number = am.annotations.shape[0]
    input_file = os.path.join(DATA, input_file)

    input_annotations = pd.read_csv("examples/valid_raw_data/annotations/input.csv")

    am.import_annotations(input_annotations)

    assert (
            am.annotations.shape[0] == input_annotations.shape[0] + original_number
    ), "first importation did not complete successfully"

    second_input = pd.read_csv(input_file)

    if exception is not None:
        with pytest.raises(exception):
            am.import_annotations(second_input, overwrite_existing=ow)
    else:
        imported, errors = am.import_annotations(second_input, overwrite_existing=ow)

        if rimported is not None:
            rimported = os.path.join(TRUTH, rimported)
            # imported.to_csv(rimported, index=False)
            rimported = pd.read_csv(rimported)
            pd.testing.assert_frame_equal(rimported.reset_index(drop=True),
                                          imported.drop(['imported_at', 'package_version'], axis=1).reset_index(
                                              drop=True),
                                          check_like=True,
                                          check_dtype=False)

        if rerrors is not None:
            rerrors = os.path.join(TRUTH, rerrors)
            # errors.to_csv(rerrors, index=False)
            rerrors = pd.read_csv(rerrors)
            pd.testing.assert_frame_equal(rerrors.reset_index(drop=True),
                                          errors.drop(['imported_at', 'package_version'], axis=1).reset_index(
                                              drop=True),
                                          check_like=True,
                                          check_dtype=False)

        am.read()
        assert all(
            [
                os.path.exists(
                    os.path.join(
                        project.path,
                        "annotations",
                        a["set"],
                        "converted",
                        a["annotation_filename"],
                    )
                )
                for a in am.annotations.to_dict(orient="records")
            ]
        ), "some annotations are missing"

        errors, warnings = am.validate()
        print(errors)
        print(warnings)
        assert len(errors) == 0 and len(warnings) == 0, "malformed annotations detected"

        errors, warnings = am.read()
        print(errors)
        print(warnings)
        assert len(errors) == 0 and len(warnings) == 0, "malformed annotation indexes detected"


# function used as a derivation function, it should throw errors if not returning dataframe or without required columns
def dv_func(a, b, x, type):
    if type == 'number':
        return 1
    elif type == 'columns':
        return pd.DataFrame([], columns=['segment_onset','segment_offset'])
    elif type == 'normal':
        return x


@pytest.mark.parametrize("exists,ow",
                         [(False, False),
                          (True, False),
                          (False, True),
                          (True, True),
                          ])
def test_derive(project, am, exists, ow):
    input_set = 'vtc_present'
    output_set = 'output'
    function = partial(dv_func, type='normal')
    am.read()

    # copy the input set to act as an existing output_set
    if exists:
        shutil.copytree(src=PATH / 'annotations' / input_set, dst=PATH / 'annotations' / output_set)
        additions = am.annotations[am.annotations['set'] == input_set].copy()
        additions['set'] = output_set
        am.annotations = pd.concat([am.annotations, additions])

    imported, errors = am.derive_annotations(input_set, output_set, function, overwrite_existing=ow)
    assert imported.shape[0] == am.annotations[am.annotations['set'] == input_set].shape[0]
    assert errors.shape[0] == 0

    truth = am.annotations[am.annotations['set'] == input_set]
    truth['merged_from'] = truth['set']
    truth['set'] = output_set
    truth['format'] = 'NA'
    cols = ['imported_at', 'package_version']
    pd.testing.assert_frame_equal(truth.drop(columns=cols).reset_index(drop=True),
                                  imported.drop(columns=cols).reset_index(drop=True))


# function used for derivation but does not hav correct signature
def bad_func(a, b):
    return b


@pytest.mark.parametrize("input_set,function,output_set,exists,ow,error",
                         [("missing", partial(dv_func, type='normal'), "output", False, False, AssertionError),
                          ("vtc_present", partial(dv_func, type='number'), "output", False, False, None),
                          ("vtc_present", partial(dv_func, type='columns'), "output", False, False, None),
                          ("vtc_present", bad_func, "output", False, False, None),
                          ("vtc_present", partial(dv_func, type='normal'), "vtc_present", False, False, AssertionError),
                          ("vtc_present", 'not_a_function', "output", False, False, ValueError),
                          ])
def test_derive_inputs(project, am, input_set, function, output_set, exists, ow, error):
    am.read()
    # copy the input set to act as an existing output_set
    if exists:
        shutil.copytree(src=PATH / 'annotations' / 'vtc_present', dst=PATH / 'annotations' / output_set)
        additions = am.annotations[am.annotations['set'] == input_set].copy()
        additions['set'] = output_set
        am.annotations = pd.concat([am.annotations, additions])

    if error:
        with pytest.raises(error):
            am.derive_annotations(input_set, output_set, function, overwrite_existing=ow)
    else:
        imported, errors = am.derive_annotations(input_set, output_set, function, overwrite_existing=ow)
        # check that 0 lines were imported because of bad input
        assert imported.shape[0] == 0
        assert errors.shape[0] == am.annotations[am.annotations['set'] == input_set].shape[0]

def test_intersect(project, am):
    input_annotations = pd.read_csv("examples/valid_raw_data/annotations/intersect.csv")
    am.import_annotations(input_annotations)

    intersection = AnnotationManager.intersection(
        am.annotations[am.annotations["set"].isin(["textgrid", "vtc_rttm"])]
    ).convert_dtypes()

    a = intersection[intersection["set"] == "textgrid"]
    b = intersection[intersection["set"] == "vtc_rttm"]

    columns = a.columns.tolist()
    columns.remove("imported_at")
    columns.remove("package_version")
    columns.remove("merged_from")

    pd.testing.assert_frame_equal(
        standardize_dataframe(a, columns).convert_dtypes(),
        standardize_dataframe(
            pd.read_csv("tests/truth/intersect_a.csv"), columns
        ).convert_dtypes(),
        check_dtype=False,
    )

    pd.testing.assert_frame_equal(
        standardize_dataframe(b, columns),
        standardize_dataframe(
            pd.read_csv("tests/truth/intersect_b.csv"), columns
        ).convert_dtypes(),
        check_dtype=False,
    )


def test_within_ranges(project, am):
    annotations = [
        {
            "recording_filename": "sound.wav",
            "set": "matching",
            "range_onset": onset,
            "range_offset": onset + 500,
        }
        for onset in np.arange(0, 4000, 500)
    ]

    matching_annotations = pd.DataFrame(
        [
            annotation
            for annotation in annotations
            if annotation["range_onset"] >= 1000 and annotation["range_offset"] <= 3000
        ]
    )

    am.annotations = pd.DataFrame(annotations)

    ranges = pd.DataFrame(
        [{"recording_filename": "sound.wav", "range_onset": 1000, "range_offset": 3000}]
    )

    matches = am.get_within_ranges(ranges, ["matching"])

    pd.testing.assert_frame_equal(
        standardize_dataframe(matching_annotations, matching_annotations.columns),
        standardize_dataframe(matches, matching_annotations.columns),
    )

    ranges["range_offset"] = 5000
    exception_caught = False
    try:
        am.get_within_ranges(ranges, ["matching"], "raise")
    except Exception as e:
        print(e)
        if str(e) == ("annotations from set 'matching' do not cover the whole selected range for recording "
                      "'sound.wav', 3.000s covered instead of 4.000s"):
            exception_caught = True

    assert (
        exception_caught
    ), "get_within_ranges should raise an exception when annotations do not fully cover the required ranges"


def test_merge(project, am):
    input_annotations = pd.read_csv("examples/valid_raw_data/annotations/input.csv")
    input_annotations = input_annotations[
        input_annotations["set"].isin(["vtc_rttm", "alice/output"])
    ]
    print(input_annotations)
    am.import_annotations(input_annotations)
    am.read()

    print(am.annotations)
    am.read()
    am.merge_sets(
        left_set="vtc_rttm",
        right_set="alice/output",
        left_columns=["speaker_type"],
        right_columns=["phonemes", "syllables", "words"],
        output_set="alice_vtc",
        full_set_merge=False,
        recording_filter={'sound.wav'}
    )
    am.read()

    anns = am.annotations[am.annotations['set'] == 'alice_vtc']
    assert anns.shape[0] == 1
    assert anns.iloc[0]['recording_filename'] == 'sound.wav'

    time.sleep(
        2)  # sleeping for 2 seconds to have different 'imported_at' values so that can make sure both merge did fine

    am.merge_sets(
        left_set="vtc_rttm",
        right_set="alice/output",
        left_columns=["speaker_type"],
        right_columns=["phonemes", "syllables", "words"],
        output_set="alice_vtc",
        full_set_merge=False,
        skip_existing=True
    )
    am.read()

    anns = am.annotations[am.annotations['set'] == 'alice_vtc']
    assert anns.shape[0] == 2
    assert set(anns['recording_filename'].unique()) == {'sound.wav', 'sound2.wav'}
    assert anns.iloc[0]['imported_at'] != anns.iloc[1]['imported_at']

    segments = am.get_segments(am.annotations[am.annotations["set"] == "alice_vtc"])
    vtc_segments = am.get_segments(am.annotations[am.annotations["set"] == "vtc_rttm"])
    assert segments.shape[0] == vtc_segments.shape[0]
    assert segments.shape[1] == vtc_segments.shape[1] + 3

    adult_segments = (
        segments[segments["speaker_type"].isin(["FEM", "MAL"])]
        .sort_values(["segment_onset", "segment_offset"])
        .reset_index(drop=True)
    )
    alice = (
        am.get_segments(am.annotations[am.annotations["set"] == "alice/output"])
        .sort_values(["segment_onset", "segment_offset"])
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        adult_segments[["phonemes", "syllables", "words"]],
        alice[["phonemes", "syllables", "words"]],
    )


def test_clipping(project, am):
    input_annotations = pd.read_csv("examples/valid_raw_data/annotations/input.csv")
    input_annotations = input_annotations[input_annotations["recording_filename"] == "sound.wav"]
    am.import_annotations(input_annotations[input_annotations["set"] == "vtc_rttm"])
    am.read()

    start = 1981000
    stop = 1984000
    segments = am.get_segments(am.annotations[am.annotations["set"] == "vtc_rttm"])
    segments = am.clip_segments(segments, start, stop)

    assert (
            segments["segment_onset"].between(start, stop).all()
            and segments["segment_offset"].between(start, stop).all()
    ), "segments not properly clipped"
    assert segments.shape[0] == 2, "got {} segments, expected 2".format(
        segments.shape[0]
    )


def test_within_time_range(project, am):
    from ChildProject.utils import TimeInterval

    am.project.recordings = pd.read_csv("tests/data/time_range_recordings.csv")

    annotations = pd.read_csv("tests/data/time_range_annotations.csv")
    matches = am.get_within_time_range(annotations, TimeInterval(datetime.datetime(1900, 1, 1, 9, 0),
                                                                 datetime.datetime(1900, 1, 1, 20, 0)))

    truth = pd.read_csv("tests/truth/time_range.csv")

    pd.testing.assert_frame_equal(
        standardize_dataframe(matches, truth.columns),
        standardize_dataframe(truth, truth.columns),
    )

    matches = am.get_within_time_range(annotations, start_time="09:00", end_time="20:00")

    pd.testing.assert_frame_equal(
        standardize_dataframe(matches, truth.columns),
        standardize_dataframe(truth, truth.columns),
    )

    exception_caught = False
    try:
        am.get_within_time_range(annotations, "9am", "8pm")
    except ValueError:
        exception_caught = True

    assert exception_caught, "no exception was thrown despite invalid times"


def test_segments_timestamps(project, am):
    segments = pd.DataFrame(
        [
            {
                "recording_filename": "sound.wav",
                "segment_onset": 3600 * 1000,
                "segment_offset": 3600 * 1000 + 1000,
            }
        ]
    )
    segments = am.get_segments_timestamps(segments)

    truth = pd.DataFrame(
        [
            {
                "recording_filename": "sound.wav",
                "segment_onset": 3600 * 1000,
                "segment_offset": 3600 * 1000 + 1000,
                "onset_time": datetime.datetime(2020, 4, 20, 9 + 1, 0, 0),
                "offset_time": datetime.datetime(2020, 4, 20, 9 + 1, 0, 1),
            }
        ]
    )

    pd.testing.assert_frame_equal(
        standardize_dataframe(segments, truth.columns),
        standardize_dataframe(truth, truth.columns),
    )


# old set, new set, error to expect, mf = add a merged from column to index, index= add a fictional index line
@pytest.mark.parametrize("old,new,error,mf,index",
                         [("textgrid", 'vtc_rttm', Exception, False, False),
                          ("invented", 'renamed', Exception, False, False),
                          ("textgrid", 'renamed', None, False, False),
                          ("textgrid", 'alice', None, False, False),  # in subdomain of alice/output
                          ("textgrid", 'invented', Exception, False, True),
                          ("textgrid", 'renamed', None, True, False),
                          ])
def test_rename(project, am, old, new, error, mf, index):
    input_annotations = pd.read_csv("examples/valid_raw_data/annotations/input.csv")
    if mf:
        am.import_annotations(input_annotations)
    else:
        am.import_annotations(input_annotations[input_annotations['set'] == old])
    am.read()

    if mf:
        mdf = pd.read_csv("examples/valid_raw_data/annotations/merged_from.csv")
        am.annotations['merged_from'] = mdf['merged_from']
        am.write()

        wanted_list = am.annotations.sort_values(['set', 'recording_filename', 'range_onset', 'range_offset'])[
            'merged_from'].astype(str).str.split(',').values.tolist()
        i = 0
        while i < len(wanted_list):
            j = 0
            while j < len(wanted_list[i]):
                if wanted_list[i][j] == old:
                    wanted_list[i][j] = new
                j += 1
            i += 1
    if index:
        add = pd.read_csv("examples/valid_raw_data/annotations/input.csv").head(1)
        add['set'] = new
        am.annotations = pd.concat([am.annotations, add])
        am.write()

    tg_count = am.annotations[am.annotations["set"] == "textgrid"].shape[0]

    if error:
        with pytest.raises(error):
            am.rename_set(old, new)
    else:
        am.rename_set(old, new)
        am.read()

        errors, warnings = am.validate()
        assert len(errors) == 0 and len(warnings) == 0, "malformed annotations detected"

        assert am.annotations[am.annotations["set"] == old].shape[0] == 0
        assert am.annotations[am.annotations["set"] == new].shape[0] == tg_count

        if mf:
            result = am.annotations.sort_values(['set', 'recording_filename', 'range_onset', 'range_offset'])[
                'merged_from'].astype(str).str.split(',').values.tolist()
            i = 0
            print(wanted_list)
            print(result)
            while i < len(wanted_list):
                assert sorted(wanted_list[i]) == sorted(result[i])
                i += 1


def custom_function(filename):
    from ChildProject.converters import VtcConverter

    df = pd.read_csv(
        filename,
        sep=" ",
        names=[
            "type",
            "file",
            "chnl",
            "tbeg",
            "tdur",
            "ortho",
            "stype",
            "name",
            "conf",
            "unk",
        ],
    )

    df["segment_onset"] = 1000 * df["tbeg"].astype(int)
    df["segment_offset"] = (1000 * (df["tbeg"] + df["tdur"])).astype(int)
    df["speaker_type"] = df["name"].map(VtcConverter.SPEAKER_TYPE_TRANSLATION)

    df.drop(
        [
            "type",
            "file",
            "chnl",
            "tbeg",
            "tdur",
            "ortho",
            "stype",
            "name",
            "conf",
            "unk",
        ],
        axis=1,
        inplace=True,
    )
    return df


def test_custom_importation(project, am):
    input = pd.DataFrame(
        [
            {
                "set": "vtc_rttm",
                "range_onset": 0,
                "range_offset": 4000,
                "recording_filename": "sound.wav",
                "time_seek": 0,
                "raw_filename": "example.rttm",
                "format": "custom",
            }
        ]
    )

    am.import_annotations(input, import_function=custom_function)
    am.read()

    errors, warnings = am.validate()
    assert len(errors) == 0


def test_set_from_path(project, am):
    assert am.set_from_path(os.path.join(project.path, "annotations/set")) == "set"
    assert am.set_from_path(os.path.join(project.path, "annotations/set/")) == "set"
    assert (
            am.set_from_path(os.path.join(project.path, "annotations/set/subset"))
            == "set/subset"
    )
    assert (
            am.set_from_path(os.path.join(project.path, "annotations/set/subset/converted"))
            == "set/subset"
    )
    assert (
            am.set_from_path(os.path.join(project.path, "annotations/set/subset/raw"))
            == "set/subset"
    )


# its
def gather_columns_to_dict(start_col, end_col, row):
    n = 1
    list_segments = []
    while True:
        start_key = "{}{}".format(start_col, n)
        end_key = "{}{}".format(end_col, n)

        if start_key in row.keys() and not pd.isnull(row[start_key]):
            list_segments.append({"start": row[start_key], "end": row[end_key]})
        else:
            return list_segments

        n += 1


def check_its(segments, truth):
    segments["cries"] = segments["cries"].astype(str)
    segments["utterances"] = (
        segments["utterances"]
        .apply(lambda line: [{"start": u["start"], "end": u["end"]} for u in line])
        .astype(str)
    )
    segments["vfxs"] = segments["vfxs"].astype(str)

    truth.rename(
        columns={
            "startTime": "segment_onset",
            "endTime": "segment_offset",
            "average_dB": "average_db",
            "peak_dB": "peak_db",
            "blkTypeId": "lena_block_number",
            "convTurnType": "lena_conv_turn_type",
            "convFloorType": "lena_conv_floor_type",
        },
        inplace=True,
    )

    truth["words"] = (
        truth[["maleAdultWordCnt", "femaleAdultWordCnt"]]
        .astype(float, errors="ignore")
        .fillna(0)
        .sum(axis=1)
    )
    truth["utterances_count"] = (
        truth[["femaleAdultUttCnt", "maleAdultUttCnt", "childUttCnt"]]
        .astype(float, errors="ignore")
        .fillna(0)
        .sum(axis=1)
    )
    truth["utterances_length"] = (
        truth[["femaleAdultUttLen", "maleAdultUttLen", "childUttLen"]]
        .astype(float, errors="ignore")
        .fillna(0)
        .sum(axis=1)
        .mul(1000)
        .astype(int)
    )
    truth["non_speech_length"] = (
        truth[["femaleAdultNonSpeechLen", "maleAdultNonSpeechLen"]]
        .astype(float, errors="ignore")
        .fillna(0)
        .sum(axis=1)
        .mul(1000)
        .astype(int)
    )

    truth["lena_block_type"] = truth.apply(
        lambda row: "pause" if row["blkType"] == "Pause" else row["convType"], axis=1
    )
    truth["lena_response_count"] = (
        truth["conversationInfo"]
        .apply(lambda s: "NA" if pd.isnull(s) else s.split("|")[1:-1][3])
        .astype(str)
    )

    truth["cries"] = truth.apply(
        partial(gather_columns_to_dict, "startCry", "endCry"), axis=1
    ).astype(str)
    truth["utterances"] = truth.apply(
        partial(gather_columns_to_dict, "startUtt", "endUtt"), axis=1
    ).astype(str)
    truth["vfxs"] = truth.apply(
        partial(gather_columns_to_dict, "startVfx", "endVfx"), axis=1
    ).astype(str)

    truth["segment_onset"] = (truth["segment_onset"] * 1000).round().astype(int)
    truth["segment_offset"] = (truth["segment_offset"] * 1000).round().astype(int)

    truth["lena_conv_floor_type"].fillna("NA", inplace=True)
    truth["lena_conv_turn_type"].fillna("NA", inplace=True)
    truth["lena_response_count"].fillna("NA", inplace=True)

    columns = [
        "segment_onset",
        "segment_offset",
        "average_db",
        "peak_db",
        "words",
        "utterances_count",
        "utterances_length",
        "non_speech_length",
        "lena_block_number",  # 'lena_block_type',
        "lena_response_count",
        "cries",
        "utterances",
        "vfxs",
        "lena_conv_turn_type",
        "lena_conv_floor_type",
    ]

    pd.testing.assert_frame_equal(
        standardize_dataframe(truth, columns),
        standardize_dataframe(segments, columns),
        check_dtype=False,
    )
