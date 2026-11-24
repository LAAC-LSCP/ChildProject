from functools import partial

import pandas as pd
import pytest

from ChildProject.converters import (
    AliceConverter,
    AnnotationConverter,
    ChatConverter,
    CsvConverter,
    EafConverter,
    Formats,
    ItsConverter,
    TextGridConverter,
    VcmConverter,
    VtcConverter,
    W2V2SMChunksConverter,
    converters,
    extensions,
)

from conftest import TEST_DATA, TRUTH


def standardize(df, columns):
    df = df[list(columns)]
    return df.sort_index(axis=1).sort_values(list(columns)).reset_index(drop=True)


def assert_same_segments(converted, truth, **kwargs):
    pd.testing.assert_frame_equal(
        standardize(converted, converted.columns), standardize(truth, converted.columns), **kwargs
    )


# ============================================================ registry
def test_every_format_has_a_registered_converter():
    assert set(converters) == {f.value for f in Formats}


def test_every_format_has_a_file_extension():
    assert set(extensions) == {f.value for f in Formats}


@pytest.mark.parametrize("fmt,cls", [
    ("csv", CsvConverter), ("vtc_rttm", VtcConverter), ("vcm_rttm", VcmConverter), ("alice", AliceConverter),
    ("its", ItsConverter), ("TextGrid", TextGridConverter), ("eaf", EafConverter), ("cha", ChatConverter),
    ("w2v2-sm", W2V2SMChunksConverter),
])
def test_converter_registration(fmt, cls):
    assert converters[fmt] is cls


def test_subclassing_registers_new_format():
    class DummyConverter(AnnotationConverter):
        FORMAT = "dummy_format_for_test"

    try:
        assert converters["dummy_format_for_test"] is DummyConverter
    finally:
        del converters["dummy_format_for_test"]


def test_only_chat_converter_is_not_thread_safe():
    assert [f for f, c in converters.items() if not c.THREAD_SAFE] == ["cha"]


def test_speaker_id_to_type_defaults_to_NA():
    assert AnnotationConverter.SPEAKER_ID_TO_TYPE["unknown_speaker"] == "NA"


# ============================================================ csv
def test_csv_converter():
    converted = CsvConverter().convert(TEST_DATA / "csv.csv").fillna("NA")
    truth = pd.read_csv(TRUTH / "csv.csv").fillna("NA")
    assert_same_segments(converted, truth, check_dtype=False)


# ============================================================ vtc
def test_vtc_converter():
    converted = VtcConverter().convert(TEST_DATA / "vtc.rttm")
    truth = pd.read_csv(TRUTH / "vtc.csv").fillna("NA")
    assert_same_segments(converted, truth, check_dtype=False)


def test_vtc_converter_warns_about_several_recordings(capsys):
    VtcConverter.convert(TEST_DATA / "vtc.rttm")
    assert "contains annotations from 2 different audio files" in capsys.readouterr().out


def test_vtc_converter_filters_on_source_file(capsys):
    converted = VtcConverter.convert(TEST_DATA / "vtc.rttm", source_file="namibie_aiku_20160714_1")
    assert converted.shape[0] == 5
    assert "WARNING" not in capsys.readouterr().out


@pytest.mark.parametrize("name,speaker_type", [("KCHI", "CHI"), ("CHI", "OCH"), ("FEM", "FEM"), ("MAL", "MAL"), ("SPEECH", "NA")])
def test_vtc_speaker_translation(name, speaker_type):
    assert VtcConverter.SPEAKER_TYPE_TRANSLATION[name] == speaker_type


# ============================================================ vcm
def test_vcm_converter():
    converted = VcmConverter().convert(TEST_DATA / "vcm.rttm")
    truth = pd.read_csv(TRUTH / "vcm.csv").fillna("NA")
    assert_same_segments(converted, truth, check_dtype=False)


def test_vcm_converter_filters_on_source_file():
    assert VcmConverter.convert(TEST_DATA / "vcm.rttm", source_file="other_recording").shape[0] == 0


# ============================================================ alice
def test_alice_converter():
    converted = AliceConverter().convert(TEST_DATA / "alice.txt")
    truth = pd.read_csv(TRUTH / "alice.csv").fillna("NA")
    assert_same_segments(converted, truth)


def test_alice_converter_filters_on_source_file():
    assert AliceConverter.convert(TEST_DATA / "alice.txt", source_file="not_this_one").shape[0] == 0


def test_alice_converter_warns_about_several_recordings(tmp_path, capsys):
    alice = tmp_path / "alice.txt"
    alice.write_text("recA_00000010_00000020.wav\t1\t1\t1\nrecB_00000010_00000020.wav\t2\t2\t2\n")
    converted = AliceConverter.convert(alice)
    assert "contains annotations from 2 different audio files" in capsys.readouterr().out
    assert converted["segment_onset"].tolist() == [1, 1]


# ============================================================ eaf
def test_eaf_converter():
    converted = EafConverter().convert(TEST_DATA / "eaf.eaf")
    truth = pd.read_csv(TRUTH / "eaf.csv", dtype={"transcription": str}).fillna("NA")
    assert_same_segments(converted, truth)


def test_eaf_converter_new_tiers():
    converted = EafConverter().convert(TEST_DATA / "eaf_any_tier.eaf", new_tiers=["newtier", "newtier2"])
    converted = converted.replace("", "NA").fillna("NA")
    truth = pd.read_csv(TRUTH / "eaf_any_tier.csv", dtype={"transcription": str}).fillna("NA")
    assert_same_segments(converted, truth)


@pytest.mark.parametrize("kwargs", [{}, {"new_tiers": None}])
def test_eaf_converter_ignores_extra_tiers_when_not_requested(kwargs):
    """without new_tiers (missing, or None as passed by AnnotationManager), dependent tiers are ignored"""
    converted = EafConverter.convert(TEST_DATA / "eaf_any_tier.eaf", **kwargs)
    assert "newtier" not in converted.columns


def test_eaf_converter_warns_about_unknown_tiers(capsys):
    EafConverter.convert(TEST_DATA / "eaf_any_tier.eaf", new_tiers=[])
    assert "warning: unknown tier" in capsys.readouterr().out


def _write_eaf(path, tiers):
    import pympi

    eaf = pympi.Elan.Eaf()
    for name, parent, annotations in tiers:
        if parent is None:
            eaf.add_tier(name)
            for start, end, value in annotations:
                eaf.add_annotation(name, start, end, value)
        else:
            eaf.add_linguistic_type(name + "_type", "Symbolic_Association")
            eaf.add_tier(name, ling=name + "_type", parent=parent)
            for time, value in annotations:
                eaf.add_ref_annotation(name, parent, time, value)
    eaf.to_file(str(path))
    return path


def test_eaf_converter_dependent_tier_column(tmp_path):
    path = _write_eaf(tmp_path / "f.eaf", [("CHI", None, [(0, 100, "hi")]), ("vcm@CHI", "CHI", [(50, "C")])])
    converted = EafConverter.convert(path)
    assert converted[["segment_onset", "segment_offset", "speaker_type", "transcription", "vcm_type"]].values.tolist() == [
        [0, 100, "CHI", "hi", "C"]
    ]


def test_eaf_converter_empty_file(tmp_path):
    path = _write_eaf(tmp_path / "empty.eaf", [("CHI", None, []), ("xds@CHI", "CHI", [])])
    converted = EafConverter.convert(path)
    assert converted.shape[0] == 0
    assert set(converted.columns) == {"segment_onset", "segment_offset", "speaker_id", "speaker_type", "addressee"}


# ============================================================ textgrid
def test_textgrid_converter():
    converted = TextGridConverter().convert(TEST_DATA / "textgrid.TextGrid")
    truth = pd.read_csv(TRUTH / "textgrid.csv", dtype={"ling_type": str}).fillna("NA")
    assert_same_segments(converted, truth)


def _write_textgrid(path, tiers):
    import pympi

    tg = pympi.Praat.TextGrid(xmax=10)
    for name, intervals in tiers:
        tier = tg.add_tier(name)
        for start, end, value in intervals:
            tier.add_interval(start, end, value)
    tg.to_file(str(path))
    return path


@pytest.mark.parametrize("value,ling_type", [("0", "0"), ("1", "1"), ("01", "NA"), ("x", "NA")])
def test_textgrid_ling_type(tmp_path, value, ling_type):
    path = _write_textgrid(tmp_path / "f.TextGrid", [("CHI", [(0, 1, value)])])
    assert TextGridConverter.convert(path)["ling_type"].tolist() == [ling_type]


def test_textgrid_skips_autre_tier_and_empty_intervals(tmp_path):
    path = _write_textgrid(tmp_path / "f.TextGrid", [("Autre", [(0, 1, "0")]), ("FA1", [(1, 2, ""), (2, 3, "1")])])
    converted = TextGridConverter.convert(path)
    assert converted[["segment_onset", "speaker_type"]].values.tolist() == [[2000, "FEM"]]


# ============================================================ chat
def test_chat_converter():
    converted = ChatConverter.convert(TEST_DATA / "vandam.cha")
    truth = pd.read_csv(TRUTH / "cha.csv").fillna("NA")
    assert_same_segments(converted, truth)


CHAT_HEADER = (
    "@UTF8\n@Begin\n@Languages:\teng\n"
    "@Participants:\tCHI Target_Child, MOT Mother\n"
    "@ID:\teng|corpus|CHI|||||Target_Child|||\n@ID:\teng|corpus|MOT|||||Mother|||\n"
)


def test_chat_converter_without_utterances(tmp_path):
    path = tmp_path / "empty.cha"
    path.write_text(CHAT_HEADER + "@End\n")
    assert ChatConverter.convert(path).shape == (0, 0)


def test_chat_converter_discards_utterances_without_timestamps(tmp_path, capsys):
    path = tmp_path / "f.cha"
    path.write_text(CHAT_HEADER + "*MOT:\thello there . \x150_1000\x15\n*CHI:\tyes .\n@End\n")
    converted = ChatConverter.convert(path)
    assert converted["speaker_id"].tolist() == ["MOT"]
    assert converted["words"].tolist() == [2]
    assert "don't have timestamps" in capsys.readouterr().out
    assert "addressee" not in converted.columns


# ============================================================ its
def gather_columns_to_dict(start_col, end_col, row):
    n = 1
    segments = []
    while True:
        start_key, end_key = f"{start_col}{n}", f"{end_col}{n}"
        if start_key in row.keys() and not pd.isnull(row[start_key]):
            segments.append({"start": row[start_key], "end": row[end_key]})
        else:
            return segments
        n += 1


def its_truth(its):
    truth = pd.read_csv(TRUTH / "its" / f"{its}_ITS_Segments.csv", dtype={"recordingInfo": "object"})
    truth = truth.rename(columns={
        "startTime": "segment_onset", "endTime": "segment_offset", "average_dB": "average_db", "peak_dB": "peak_db",
        "blkTypeId": "lena_block_number", "convTurnType": "lena_conv_turn_type", "convFloorType": "lena_conv_floor_type",
    })

    def total(cols, scale=None):
        s = truth[cols].astype(float, errors="ignore").fillna(0).sum(axis=1)
        return s.mul(scale).astype(int) if scale else s

    truth["words"] = total(["maleAdultWordCnt", "femaleAdultWordCnt"])
    truth["utterances_count"] = total(["femaleAdultUttCnt", "maleAdultUttCnt", "childUttCnt"])
    truth["utterances_length"] = total(["femaleAdultUttLen", "maleAdultUttLen", "childUttLen"], 1000)
    truth["non_speech_length"] = total(["femaleAdultNonSpeechLen", "maleAdultNonSpeechLen"], 1000)
    truth["lena_response_count"] = truth["conversationInfo"].apply(
        lambda s: "NA" if pd.isnull(s) else s.split("|")[1:-1][3]).astype(str)
    for name, (start, end) in {"cries": ("startCry", "endCry"), "utterances": ("startUtt", "endUtt"),
                               "vfxs": ("startVfx", "endVfx")}.items():
        truth[name] = truth.apply(partial(gather_columns_to_dict, start, end), axis=1).astype(str)
    truth["segment_onset"] = (truth["segment_onset"] * 1000).round().astype(int)
    truth["segment_offset"] = (truth["segment_offset"] * 1000).round().astype(int)
    for col in ["lena_conv_floor_type", "lena_conv_turn_type", "lena_response_count"]:
        truth[col] = truth[col].fillna("NA")
    return truth


ITS_COLUMNS = ["segment_onset", "segment_offset", "average_db", "peak_db", "words", "utterances_count",
               "utterances_length", "non_speech_length", "lena_block_number", "lena_response_count", "cries",
               "utterances", "vfxs", "lena_conv_turn_type", "lena_conv_floor_type"]


@pytest.mark.parametrize("its", ["example_lena_new", "example_lena_old"])
def test_its_converter(its):
    converted = ItsConverter().convert(TEST_DATA / f"{its}.its")
    converted["cries"] = converted["cries"].astype(str)
    converted["utterances"] = converted["utterances"].apply(
        lambda line: [{"start": u["start"], "end": u["end"]} for u in line]).astype(str)
    converted["vfxs"] = converted["vfxs"].astype(str)
    pd.testing.assert_frame_equal(standardize(its_truth(its), ITS_COLUMNS), standardize(converted, ITS_COLUMNS),
                                  check_dtype=False)


def test_its_converter_unknown_recording_number():
    converted = ItsConverter.convert(TEST_DATA / "example_lena_new.its", recording_num=999)
    assert converted.shape[0] == 0


# ============================================================ w2v2 speech maturity chunks
def test_w2v2_id_split_returns_recording_onset_offset():
    """documented as returning 'audio filename without extension, onset, offset'"""
    assert W2V2SMChunksConverter.id_split("lena_rec01_1000_2000") == ("lena_rec01", 1000, 2000)


@pytest.fixture
def w2v2_csv(tmp_path):
    path = tmp_path / "chunks.csv"
    pd.DataFrame({
        "id": ["recA_1000_2000", "recA_3000_3500", "lena_recB_0_500"],
        "prediction_class_name": ["Canonical", "Crying", "Junk"],
    }).to_csv(path, index=False)
    return path


def test_w2v2_converter(w2v2_csv):
    converted = W2V2SMChunksConverter.convert(w2v2_csv)
    assert converted.values.tolist() == [[1000, 2000, "C"], [3000, 3500, "Y"], [0, 500, "J"]]


def test_w2v2_converter_filters_on_recording(w2v2_csv):
    converted = W2V2SMChunksConverter.convert(w2v2_csv, recording="lena/recB")
    assert converted.values.tolist() == [[0, 500, "J"]]
