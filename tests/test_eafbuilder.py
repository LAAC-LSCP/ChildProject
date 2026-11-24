"""ELAN template generation (ChildProject.pipelines.eafbuilder)"""
import os
from importlib import resources

import pandas as pd
import pytest
from pympi import Eaf

from ChildProject.pipelines.eafbuilder import CHILDPROJECT_TYPE, EafBuilderPipeline, create_eaf
from ChildProject.pipelines.samplers import PeriodicSampler

from conftest import TEST_DATA, import_static

with resources.path("ChildProject.templates", "basic.etf") as _etf:
    TEMPLATE = str(_etf)

SPEECH = pd.DataFrame({"segment_onset": [10, 30, 50], "segment_offset": [15, 35, 55],
                       "speaker_type": ["FEM", "CHI", None]})


def annotations(eaf, tier):
    return sorted((int(eaf.timeslots[a]), int(eaf.timeslots[b]), value) for a, b, value, _ in eaf.tiers[tier][0].values())


def build(tmp_path, recording="sound.wav", timestamps=((10, 20),), **kwargs):
    args = dict(contxt_on=0, contxt_off=0, speech_segments=None, imported_set=None, imported_format=None)
    args.update(kwargs)
    create_eaf(TEMPLATE, "out", str(tmp_path), recording, list(timestamps), "periodic", args.pop("contxt_on"),
               args.pop("contxt_off"), **args)
    return Eaf(str(tmp_path / "out.eaf"))


# ============================================================ create_eaf
def test_create_eaf_writes_file(tmp_path):
    build(tmp_path)
    assert (tmp_path / "out.eaf").exists()


def test_create_eaf_sampler_tiers(tmp_path):
    eaf = build(tmp_path)
    assert {"SAMPLER", "code_periodic", "context_periodic", "code_num_periodic"} <= set(eaf.tiers)
    assert eaf.tiers["code_periodic"][2]["PARENT_REF"] == "SAMPLER"
    assert CHILDPROJECT_TYPE in eaf.linguistic_types


def test_create_eaf_code_segments(tmp_path):
    eaf = build(tmp_path, timestamps=[(10, 20), (30, 40)])
    assert annotations(eaf, "code_periodic") == [(10, 20, ""), (30, 40, "")]
    assert annotations(eaf, "code_num_periodic") == [(10, 20, "periodic1"), (30, 40, "periodic2")]


def test_create_eaf_context(tmp_path):
    eaf = build(tmp_path, timestamps=[(100, 200)], contxt_on=30, contxt_off=40)
    assert annotations(eaf, "context_periodic") == [(70, 240, "")]


def test_create_eaf_context_starts_at_zero_at_most(tmp_path):
    eaf = build(tmp_path, timestamps=[(10, 20)], contxt_on=100)
    assert annotations(eaf, "context_periodic") == [(0, 20, "")]


@pytest.mark.parametrize("recording,mimetype", [
    ("sound.wav", "audio/x-wav"),
    ("video.mp4", "video/mp4"),
    ("audio.mp3", "audio/mpeg"),
    ("audio.FLAC", "audio/x-flac"),
])
def test_create_eaf_links_media_with_mimetype(tmp_path, recording, mimetype):
    eaf = build(tmp_path, recording=recording)
    assert eaf.media_descriptors[0]["MEDIA_URL"] == recording
    assert eaf.media_descriptors[0]["MIME_TYPE"] == mimetype


def test_create_eaf_speech_tiers_named_after_set(tmp_path):
    eaf = build(tmp_path, speech_segments=SPEECH, imported_set="vtc", imported_format="csv")
    assert annotations(eaf, "VTC-FEM") == [(10, 15, "")]
    assert annotations(eaf, "VTC-CHI") == [(30, 35, "")]


def test_create_eaf_missing_speaker_type_is_speech_for_vtc(tmp_path):
    eaf = build(tmp_path, speech_segments=SPEECH, imported_set="vtc", imported_format="vtc_rttm")
    assert annotations(eaf, "VTC-SPEECH") == [(50, 55, "")]


def test_create_eaf_missing_speaker_type_is_skipped_for_other_formats(tmp_path):
    eaf = build(tmp_path, speech_segments=SPEECH, imported_set="vtc", imported_format="eaf")
    assert not any("SPEECH" in t or "NAN" in t.upper() for t in eaf.tiers)


def test_create_eaf_nested_set_name(tmp_path):
    eaf = build(tmp_path, speech_segments=SPEECH.iloc[:1], imported_set="vtc/sub")
    assert "VTC_SUB-FEM" in eaf.tiers


def test_create_eaf_without_set_name(tmp_path):
    eaf = build(tmp_path, speech_segments=SPEECH.iloc[:1])
    assert "FEM" in eaf.tiers


def test_create_eaf_prefers_speaker_id(tmp_path):
    eaf = build(tmp_path, speech_segments=SPEECH.iloc[:1].assign(speaker_id="MOT"))
    assert "MOT" in eaf.tiers and "FEM" not in eaf.tiers


def test_create_eaf_segments_without_speaker_are_skipped(tmp_path):
    eaf = build(tmp_path, speech_segments=SPEECH[["segment_onset", "segment_offset"]])
    assert set(eaf.tiers) <= {"SAMPLER", "code_periodic", "context_periodic", "code_num_periodic"} | set(Eaf(TEMPLATE).tiers)


def test_create_eaf_invalid_template(tmp_path):
    with pytest.raises(Exception):
        create_eaf("README.md", "out", str(tmp_path), "sound.wav", [], "periodic", 0, 0)


def test_create_eaf_invalid_timestamps(tmp_path):
    with pytest.raises(ValueError):
        create_eaf(TEMPLATE, "out", str(tmp_path), "sound.wav", [(5, "abc")], "periodic", 0, 0)


# ============================================================ EafBuilderPipeline
@pytest.fixture
def segments_csv(tmp_path):
    path = tmp_path / "segments.csv"
    pd.DataFrame([["sound.wav", 0, 500], ["sound.wav", 1000, 1500], ["sub/rec2.wav", 0, 100]],
                 columns=["recording_filename", "segment_onset", "segment_offset"]).to_csv(path, index=False)
    return path


def run_builder(tmp_path, segments, **kwargs):
    EafBuilderPipeline().run(destination=str(tmp_path / "eaf"), segments=str(segments), eaf_type="random",
                             template=kwargs.pop("template", "basic"), **kwargs)
    return tmp_path / "eaf"


def test_builder_one_eaf_and_pfsx_per_recording(tmp_path, segments_csv):
    out = run_builder(tmp_path, segments_csv)
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.*")) == [
        "sound_random_basic.eaf", "sound_random_basic.pfsx", "sub/rec2_random_basic.eaf", "sub/rec2_random_basic.pfsx"]


def test_builder_codes_every_segment(tmp_path, segments_csv):
    out = run_builder(tmp_path, segments_csv)
    eaf = Eaf(str(out / "sound_random_basic.eaf"))
    assert annotations(eaf, "code_random") == [(0, 500, ""), (1000, 1500, "")]


def test_builder_context(tmp_path, segments_csv):
    out = run_builder(tmp_path, segments_csv, context_onset=100, context_offset=200)
    assert annotations(Eaf(str(out / "sound_random_basic.eaf")), "context_random") == [(0, 700, ""), (900, 1700, "")]


@pytest.mark.parametrize("template", ["basic", "native", "non-native"])
def test_builder_bundled_templates(tmp_path, segments_csv, template):
    out = run_builder(tmp_path, segments_csv, template=template)
    assert (out / f"sound_random_{template}.pfsx").exists()


def test_builder_custom_template(tmp_path, segments_csv):
    with resources.path("ChildProject.templates", "basic.pfsx") as pfsx:
        (tmp_path / "mine.etf").write_bytes(open(TEMPLATE, "rb").read())
        (tmp_path / "mine.pfsx").write_bytes(open(pfsx, "rb").read())
    out = run_builder(tmp_path, segments_csv, template=str(tmp_path / "mine"))
    assert (out / "sound_random_mine.eaf").exists()


@pytest.mark.parametrize("missing", ["etf", "pfsx"])
def test_builder_missing_template_file(tmp_path, segments_csv, missing):
    with resources.path("ChildProject.templates", "basic.pfsx") as pfsx:
        if missing != "etf":
            (tmp_path / "mine.etf").write_bytes(open(TEMPLATE, "rb").read())
        if missing != "pfsx":
            (tmp_path / "mine.pfsx").write_bytes(open(pfsx, "rb").read())
    with pytest.raises(Exception, match=f"mine.{missing} cannot be found"):
        run_builder(tmp_path, segments_csv, template=str(tmp_path / "mine"))


def test_builder_requires_segments(tmp_path):
    with pytest.raises(AssertionError, match="should not be empty"):
        EafBuilderPipeline().extract(str(tmp_path), pd.DataFrame(columns=["recording_filename", "segment_onset", "segment_offset"]),
                                     "random", "basic")


def test_builder_requires_columns(tmp_path):
    from ChildProject.tables import MissingColumnsException
    with pytest.raises(MissingColumnsException):
        EafBuilderPipeline().extract(str(tmp_path), pd.DataFrame({"recording_filename": ["sound.wav"]}), "random", "basic")


@pytest.fixture
def prefilled(project, tmp_path):
    from ChildProject.annotations import AnnotationManager
    import_static(AnnotationManager(project), pd.read_csv(TEST_DATA / "eaf_segments.csv"), "vtc", range_offset=4000,
                  fmt="vtc_rttm")
    sampler = PeriodicSampler(project, 500, 500, 250, recordings=["sound.wav"])
    sampler.sample()
    segments_path = tmp_path / "segments.csv"
    sampler.segments.to_csv(segments_path)
    EafBuilderPipeline().run(destination=str(tmp_path / "eaf"), segments=str(segments_path), eaf_type="periodic",
                             template="basic", context_onset=250, context_offset=250, path=str(project.path),
                             import_speech_from="vtc")
    return Eaf(str(tmp_path / "eaf" / "sound_periodic_basic.eaf")), sampler.segments, project


def test_builder_prefill_codes_sampled_segments(prefilled):
    eaf, sampled, _ = prefilled
    assert [a[:2] for a in annotations(eaf, "code_periodic")] == sorted(map(tuple, sampled[["segment_onset", "segment_offset"]].values.tolist()))


@pytest.mark.parametrize("speaker,tier", [("CHI", "VTC-CHI"), ("OCH", "VTC-OCH"), ("FEM", "VTC-FEM"), (None, "VTC-SPEECH")])
def test_builder_prefill_speech_tiers(prefilled, speaker, tier):
    from ChildProject.annotations import AnnotationManager
    eaf, sampled, project = prefilled
    am = AnnotationManager(project)
    ranges = sampled.rename(columns={"segment_onset": "range_onset", "segment_offset": "range_offset"})
    segments = am.get_segments(am.get_within_ranges(ranges, ["vtc"], "warn"))
    expected = segments[segments["speaker_type"].isnull()] if speaker is None else segments[segments["speaker_type"] == speaker]
    assert [a[:2] for a in annotations(eaf, tier)] == sorted(map(tuple, expected[["segment_onset", "segment_offset"]].astype(int).values.tolist()))


def test_builder_prefill_skips_recordings_without_annotations(project, tmp_path):
    from ChildProject.annotations import AnnotationManager
    import_static(AnnotationManager(project), pd.read_csv(TEST_DATA / "eaf_segments.csv"), "vtc", range_offset=4000)
    segments = pd.DataFrame([["sound2.wav", 0, 500]], columns=["recording_filename", "segment_onset", "segment_offset"])
    EafBuilderPipeline().extract(str(tmp_path / "eaf"), segments, "periodic", "basic", path=str(project.path),
                                 import_speech_from="vtc")
    assert not (tmp_path / "eaf").exists()


def test_setup_parser():
    import argparse
    parser = argparse.ArgumentParser()
    EafBuilderPipeline.setup_parser(parser)
    args = parser.parse_args(["--segments", "s.csv", "--eaf-type", "random", "--template", "basic"])
    assert (args.context_onset, args.context_offset, args.path, args.import_speech_from) == (0, 0, None, None)
