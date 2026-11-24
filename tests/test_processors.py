"""audio processing pipelines (ChildProject.pipelines.processors)"""
import logging
import os

import numpy as np
import pandas as pd
import pytest
import soundfile
import yaml

from ChildProject.pipelines.processors import (
    AudioProcessingPipeline,
    AudioStandard,
    BasicProcessor,
    ChannelMapper,
    VettingProcessor,
    pipelines,
)
from ChildProject.projects import CONVERTED_RECORDINGS, RAW_RECORDINGS

LOGGER = "ChildProject.pipelines.processors"


def converted(project, profile):
    return pd.read_csv(project.path / CONVERTED_RECORDINGS / profile / "recordings.csv")


def profile_file(project, profile, name):
    return project.path / CONVERTED_RECORDINGS / profile / name


def basic(project, name="test", **kwargs):
    return BasicProcessor(project, name, format="wav", codec="pcm_s16le", sampling=16000, **kwargs)


@pytest.fixture
def stereo(project):
    """replace sound2.wav by a 1 s stereo file whose channels are 0.5 and -0.25"""
    data = np.stack([np.full(16000, 0.5), np.full(16000, -0.25)], axis=1)
    soundfile.write(project.path / RAW_RECORDINGS / "sound2.wav", data, samplerate=16000)
    return project


def test_pipelines_registry():
    assert pipelines == {"basic": BasicProcessor, "vetting": VettingProcessor, "channel-mapping": ChannelMapper,
                         "standard": AudioStandard}


def test_media_types():
    assert (BasicProcessor.MEDIA_TYPES, AudioStandard.MEDIA_TYPES) == (None, None)
    assert VettingProcessor.MEDIA_TYPES == ChannelMapper.MEDIA_TYPES == ["audio"]


# ============================================================ base processor
def test_output_directory(project):
    assert basic(project, "out").output_directory() == os.path.join(project.path, CONVERTED_RECORDINGS, "out")


def test_unknown_input_profile(project):
    with pytest.raises(AssertionError, match="provided input profile nope does not exist"):
        basic(project, input_profile="nope")


def test_read_metadata_without_previous_conversion(project):
    assert basic(project).read_metadata() is None


def test_process_writes_index(project):
    basic(project).process("params.yml")
    index = converted(project, "test")
    assert index["original_filename"].tolist() == ["sound.wav", "video.mp4", "sound2.wav"]
    assert (index["parameters"] == "params.yml").all()


def test_process_keeps_previous_conversions(project):
    basic(project, recordings=["sound.wav"]).process("first.yml")
    basic(project, recordings=["sound2.wav"]).process("second.yml")
    index = converted(project, "test").set_index("original_filename")
    assert index.loc["sound.wav", "parameters"] == "first.yml"
    assert index.loc["sound2.wav", "parameters"] == "second.yml"


def test_process_replaces_reconverted_recordings(project):
    basic(project, recordings=["sound.wav"]).process("first.yml")
    basic(project, recordings=["sound.wav"]).process("second.yml")
    assert converted(project, "test")["parameters"].tolist() == ["second.yml"]


def test_process_with_pool(project):
    basic(project, threads=2).process("p.yml")
    assert converted(project, "test")["success"].all()


def test_process_with_all_cores(project):
    """--threads help: '0 = uses all available cores'"""
    basic(project, threads=0).process("p.yml")
    assert converted(project, "test").shape[0] == 3


def test_process_skips_unsupported_media_types(project, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        VettingProcessor(project, "vetted", segments_path=_vetting_segments(project)).process("p.yml")
    assert "1 recordings were" in caplog.text
    assert "video.mp4" not in converted(project, "vetted")["original_filename"].tolist()


def test_process_without_supported_recordings(project):
    processor = VettingProcessor(project, "vetted", segments_path=_vetting_segments(project), recordings=["video.mp4"])
    assert processor.process("p.yml") is None


# ============================================================ basic
def test_basic_converts_every_recording(project):
    basic(project).process("p.yml")
    index = converted(project, "test")
    assert index["success"].all()
    for name in index["converted_filename"]:
        assert soundfile.info(profile_file(project, "test", name)).samplerate == 16000


def test_basic_extracts_audio_from_video(project):
    basic(project, recordings=["video.mp4"]).process("p.yml")
    info = soundfile.info(profile_file(project, "test", "video.wav"))
    assert info.duration == pytest.approx(10.0, abs=0.1)
    assert info.channels == 2


def test_basic_keeps_duration(project):
    basic(project).process("p.yml")
    durations = project.compute_recordings_duration(profile="test")
    assert durations.set_index("recording_filename").loc["sound.wav", "duration"] == 4000


def test_basic_reports_failures(project):
    (project.path / RAW_RECORDINGS / "sound2.wav").write_bytes(b"not audio")
    basic(project, recordings=["sound2.wav"]).process("p.yml")
    index = converted(project, "test")
    assert index["success"].tolist() == [False]
    assert isinstance(index["error"].iloc[0], str) and index["error"].iloc[0]


def test_basic_skip_existing(project):
    basic(project, recordings=["sound.wav"]).process("first.yml")
    result = basic(project, recordings=["sound.wav"], skip_existing=True).process_recording({"recording_filename": "sound.wav"})
    assert result.shape[0] == 0


def test_basic_skips_NA_recording(project):
    assert basic(project).process_recording({"recording_filename": "NA"}).shape[0] == 0


def test_basic_from_input_profile(project):
    basic(project, "first").process("p.yml")
    basic(project, "second", input_profile="first").process("p.yml")
    assert converted(project, "second")["success"].all()


# ============================================================ standard
def test_standard_profile(project):
    AudioStandard(project).process("p.yml")
    index = converted(project, "standard")
    assert index["success"].astype(str).eq("True").all()
    for name in index["converted_filename"]:
        info = soundfile.info(profile_file(project, "standard", name))
        assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16")


def test_standard_output_directory(project):
    assert AudioStandard(project).output_directory().endswith(os.path.join(str(CONVERTED_RECORDINGS), "standard"))


def test_standard_reports_failures(project):
    (project.path / RAW_RECORDINGS / "sound2.wav").write_bytes(b"not audio")
    AudioStandard(project, recordings=["sound2.wav"]).process("p.yml")
    assert converted(project, "standard")["success"].astype(str).tolist() == ["False"]


def test_standard_skip_existing(project):
    AudioStandard(project, recordings=["sound.wav"]).process("p.yml")
    assert AudioStandard(project, skip_existing=True).process_recording({"recording_filename": "sound.wav"}).shape[0] == 0


def test_standard_skips_NA_recording(project):
    assert AudioStandard(project).process_recording({"recording_filename": "NA"}).shape[0] == 0


# ============================================================ vetting
def _vetting_segments(project):
    path = project.path / "segments.csv"
    pd.DataFrame([{"recording_filename": "sound.wav", "segment_onset": 1000, "segment_offset": 3000},
                  {"recording_filename": "sound2.wav", "segment_onset": 0, "segment_offset": 500}]).to_csv(path)
    return path


def test_vetting_silences_segments(project):
    VettingProcessor(project, "vetted", segments_path=_vetting_segments(project), recordings=["sound.wav"]).process("p.yml")
    signal, sr = soundfile.read(profile_file(project, "vetted", "sound.wav"))
    assert np.all(signal[1 * sr:3 * sr] == 0)
    assert np.any(signal[:1 * sr] != 0)


def test_vetting_stereo(stereo):
    VettingProcessor(stereo, "vetted", segments_path=_vetting_segments(stereo), recordings=["sound2.wav"]).process("p.yml")
    signal, sr = soundfile.read(profile_file(stereo, "vetted", "sound2.wav"))
    assert signal.shape == (16000, 2)
    assert np.all(signal[:8000] == 0)
    assert signal[-1].tolist() == pytest.approx([0.5, -0.25], abs=1e-3)


def test_vetting_writes_index(project):
    VettingProcessor(project, "vetted", segments_path=_vetting_segments(project), recordings=["sound.wav"]).process("p.yml")
    assert converted(project, "vetted")[["original_filename", "converted_filename", "success"]].values.tolist() == [
        ["sound.wav", "sound.wav", True]]


# ============================================================ channel mapping
def test_channel_mapping_stereo_to_mono(stereo):
    ChannelMapper(stereo, "mapped", channels=["1,1"], recordings=["sound2.wav"]).process("p.yml")
    signal, _ = soundfile.read(profile_file(stereo, "mapped", "sound2.wav"))
    assert signal.ndim == 1
    assert signal[0] == pytest.approx(0.25, abs=1e-3)


def test_channel_mapping_swaps_channels(stereo):
    ChannelMapper(stereo, "mapped", channels=["0,1", "1,0"], recordings=["sound2.wav"]).process("p.yml")
    signal, _ = soundfile.read(profile_file(stereo, "mapped", "sound2.wav"))
    assert signal[0].tolist() == pytest.approx([-0.25, 0.5], abs=1e-3)


def test_channel_mapping_channel_mismatch(stereo, caplog):
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        ChannelMapper(stereo, "mapped", channels=["1,1,1"], recordings=["sound2.wav"]).process("p.yml")
    assert "channel mismatch (expected 3 channels, got 2)" in caplog.text
    assert converted(stereo, "mapped")["success"].tolist() == [False]


def test_channel_mapping_mono(project):
    ChannelMapper(project, "mapped", channels=["2"], recordings=["sound.wav"]).process("p.yml")
    original, _ = soundfile.read(project.path / RAW_RECORDINGS / "sound.wav")
    signal, _ = soundfile.read(profile_file(project, "mapped", "sound.wav"))
    assert signal == pytest.approx(np.clip(2 * original, -1, 1), abs=1e-3)


def test_channel_mapping_from_vetted_profile(stereo):
    VettingProcessor(stereo, "vetted", segments_path=_vetting_segments(stereo), recordings=["sound2.wav"]).process("p.yml")
    ChannelMapper(stereo, "mapped", channels=["1,1"], input_profile="vetted", recordings=["sound2.wav"]).process("p.yml")
    signal, _ = soundfile.read(profile_file(stereo, "mapped", "sound2.wav"))
    assert np.all(signal[:8000] == 0)
    assert signal[-1] == pytest.approx(0.25, abs=1e-3)


# ============================================================ AudioProcessingPipeline
def test_pipeline_run_returns_index_and_parameters(project):
    index, parameters = AudioProcessingPipeline().run(processor="basic", path=project.path, name="test", format="wav",
                                                      codec="pcm_s16le", sampling=16000)
    assert pd.read_csv(index).shape[0] == 3
    content = yaml.unsafe_load(open(parameters))
    assert {"processor": "basic"} in content["parameters"]
    assert {"sampling": 16000} in content["parameters"]


def test_pipeline_run_standard(project):
    index, _ = AudioProcessingPipeline().run(processor="standard", path=project.path)
    assert os.path.dirname(index) == os.path.join(project.path, CONVERTED_RECORDINGS, "standard")


def test_pipeline_unknown_processor(project):
    with pytest.raises(NotImplementedError, match="invalid pipeline 'nope'"):
        AudioProcessingPipeline().run(processor="nope", path=project.path)
