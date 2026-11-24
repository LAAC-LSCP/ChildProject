"""samplers (ChildProject.pipelines.samplers)"""
import logging
import os

import pandas as pd
import pytest
import yaml
from pydub import AudioSegment

from ChildProject.pipelines.samplers import (
    ConversationSampler,
    CustomSampler,
    EnergyDetectionSampler,
    HighVolubilitySampler,
    PeriodicSampler,
    RandomVocalizationSampler,
    SamplerPipeline,
    pipelines,
)

from conftest import TRUTH, import_static

AUDIO = ["sound.wav", "sound2.wav"]


def test_pipelines_registry():
    assert pipelines == {
        "custom": CustomSampler, "periodic": PeriodicSampler, "random-vocalizations": RandomVocalizationSampler,
        "energy-detection": EnergyDetectionSampler, "high-volubility": HighVolubilitySampler,
        "conversations": ConversationSampler,
    }


# ============================================================ base Sampler: exclusion and validation
EXCLUDED = pd.DataFrame([["sound.wav", 250, 750], ["sound.wav", 2000, 4000]],
                        columns=["recording_filename", "segment_onset", "segment_offset"])


def periodic(project, **kwargs):
    return PeriodicSampler(project=project, length=1000, period=1000, recordings=AUDIO, **kwargs)


def test_exclusion_cuts_sampled_segments(project):
    segments = periodic(project, exclude=EXCLUDED).sample()
    sound = segments[segments["recording_filename"] == "sound.wav"]
    assert sound[["segment_onset", "segment_offset"]].values.tolist() == [[0, 250], [750, 1000]]


def test_exclusion_keeps_other_recordings(project):
    segments = periodic(project, exclude=EXCLUDED).sample()
    sound2 = segments[segments["recording_filename"] == "sound2.wav"]
    assert sound2[["segment_onset", "segment_offset"]].values.tolist() == [[0, 1000], [2000, 3000]]


def test_exclusion_from_csv(project, tmp_path):
    EXCLUDED.to_csv(tmp_path / "excluded.csv", index=False)
    segments = periodic(project, exclude=str(tmp_path / "excluded.csv")).sample()
    assert segments[segments["recording_filename"] == "sound.wav"].shape[0] == 2


def test_exclusion_requires_columns(project):
    with pytest.raises(ValueError, match="exclude dataframe is missing"):
        periodic(project, exclude=EXCLUDED.drop(columns=["segment_onset"]))


def test_no_exclusion_keeps_segments(project):
    sampler = periodic(project)
    assert sampler.remove_excluded() is None
    assert sampler.sample().shape[0] == 4


def test_assert_valid(project):
    sampler = periodic(project)
    sampler.sample()
    assert sampler.assert_valid() is True


def test_assert_valid_missing_columns(project):
    sampler = periodic(project)
    sampler.segments = pd.DataFrame({"recording_filename": ["sound.wav"]})
    with pytest.raises(Exception, match="custom segments are missing the following columns"):
        sampler.assert_valid()


# ============================================================ custom sampler and audio export
@pytest.fixture
def custom_segments(tmp_path):
    path = tmp_path / "segments.csv"
    pd.DataFrame([["sound.wav", 0, 1000], ["video.mp4", 500, 2500]],
                 columns=["recording_filename", "segment_onset", "segment_offset"]).to_csv(path, index=False)
    return path


def test_custom_sampler(project, custom_segments):
    segments = CustomSampler(project, custom_segments).sample()
    assert segments[["recording_filename", "segment_onset"]].values.tolist() == [["sound.wav", 0], ["video.mp4", 500]]


def test_export_audio(project, custom_segments, tmp_path):
    sampler = CustomSampler(project, custom_segments)
    sampler.sample()
    sampler.export_audio(tmp_path / "chunks", format="wav")
    exported = AudioSegment.from_file(tmp_path / "chunks" / "sound_0_1000.wav")
    assert len(exported) == 1000


def test_export_audio_from_video(project, custom_segments, tmp_path):
    sampler = CustomSampler(project, custom_segments)
    sampler.sample()
    sampler.export_audio(tmp_path / "chunks", format="wav")
    assert len(AudioSegment.from_file(tmp_path / "chunks" / "video_500_2500.wav")) == 2000


def test_export_audio_requires_valid_segments(project, tmp_path):
    sampler = CustomSampler(project, "unused.csv")
    with pytest.raises(Exception, match="missing the following columns"):
        sampler.export_audio(tmp_path)


# ============================================================ periodic
@pytest.mark.parametrize("by,truth", [("recording_filename", "periodic_rec.csv"), ("session_id", "periodic_sess.csv")])
def test_periodic(project, by, truth):
    sampler = PeriodicSampler(project=project, offset=1000, length=500, period=200, recordings=AUDIO, by=by)
    sampler.sample()
    pd.testing.assert_frame_equal(sampler.segments.reset_index(drop=True), pd.read_csv(TRUTH / "sampler" / truth),
                                  check_like=True)


def test_periodic_samples_video(project):
    segments = PeriodicSampler(project=project, length=1000, period=4000, recordings=["video.mp4"]).sample()
    # video.mp4 lasts 10120 ms: the last segment is clipped to the end of the recording
    assert segments[["segment_onset", "segment_offset"]].values.tolist() == [[0, 1000], [5000, 6000], [10000, 10120]]


def test_periodic_clips_last_segment_to_recording(project):
    segments = PeriodicSampler(project=project, length=1500, period=1000, recordings=["sound.wav"]).sample()
    assert segments[["segment_onset", "segment_offset"]].values.tolist() == [[0, 1500], [2500, 4000]]


def test_periodic_computes_missing_durations(project, caplog):
    project.recordings = project.recordings.drop(columns=["duration"])
    with caplog.at_level(logging.ERROR, logger="ChildProject.pipelines.samplers"):
        segments = PeriodicSampler(project=project, length=1000, period=1000, recordings=["sound.wav"]).sample()
    assert "recordings duration was not found" in caplog.text
    assert segments.shape[0] == 2


def test_periodic_recording_without_start_time(project):
    project.recordings.loc[project.recordings["recording_filename"] == "sound.wav", "start_time"] = "NA"
    segments = PeriodicSampler(project=project, length=1000, period=1000, recordings=["sound.wav"]).sample()
    assert segments[["segment_onset", "segment_offset"]].values.tolist() == [[0, 1000], [2000, 3000]]


# ============================================================ random vocalizations
@pytest.fixture
def vocalizations(am):
    segments = pd.DataFrame({"segment_onset": [1000] * 3, "segment_offset": [2000] * 3, "speaker_type": ["CHI", "FEM", "MAL"]})
    import_static(am, segments, "random", range_offset=4000)
    return am.project


def test_random_vocalizations_targets_speaker(vocalizations):
    segments = RandomVocalizationSampler(vocalizations, "random", ["CHI"], 1, recordings=["sound.wav"]).sample()
    assert segments[["segment_onset", "segment_offset", "speaker_type"]].values.tolist() == [[1000, 2000, "CHI"]]


def test_random_vocalizations_sample_size(vocalizations):
    segments = RandomVocalizationSampler(vocalizations, "random", ["CHI", "FEM", "MAL"], 2, recordings=["sound.wav"]).sample()
    assert segments.shape[0] == 2


def test_random_vocalizations_by_child(vocalizations):
    segments = RandomVocalizationSampler(vocalizations, "random", ["FEM"], 5, by="child_id").sample()
    assert segments["speaker_type"].tolist() == ["FEM"]


def test_random_vocalizations_with_pool(vocalizations):
    segments = RandomVocalizationSampler(vocalizations, "random", ["CHI"], 1, threads=2, recordings=["sound.wav"]).sample()
    assert segments.shape[0] == 1


def test_random_vocalizations_recording_without_annotations(vocalizations, caplog):
    with caplog.at_level(logging.WARNING, logger="ChildProject.pipelines.samplers"):
        segments = RandomVocalizationSampler(vocalizations, "random", ["CHI"], 1).sample()
    assert "no annotations from the set 'random' were found for the recording 'sound2.wav'" in caplog.text
    assert segments["recording_filename"].tolist() == ["sound.wav"]


# ============================================================ energy detection
def energy(project, **kwargs):
    return EnergyDetectionSampler(project=project, windows_length=100, windows_spacing=100, windows_count=1,
                                  low_freq=200, high_freq=1000, threads=1, **kwargs)


def test_energy_detection_by_recording(project):
    segments = energy(project, threshold=0.98, recordings=AUDIO).sample()
    assert sorted(segments[["recording_filename", "segment_onset", "segment_offset"]].values.tolist()) == [
        ["sound.wav", 1900, 2000], ["sound2.wav", 1900, 2000]]


def test_energy_detection_by_child(project):
    segments = energy(project, threshold=0.99, by="child_id", recordings=AUDIO).sample()
    assert segments[["recording_filename", "segment_onset", "segment_offset"]].values.tolist() == [["sound2.wav", 1900, 2000]]


def test_energy_detection_on_video(project):
    windows = energy(project).get_recording_windows(project.recordings.set_index("recording_filename").loc["video.mp4"].to_dict() | {"recording_filename": "video.mp4"})
    assert windows.shape[0] == 100
    assert {"channel_0", "channel_1"} <= set(windows.columns)


def test_energy_detection_whole_dataset_including_video(project):
    segments = energy(project, threshold=0.98).sample()
    assert sorted(segments["recording_filename"]) == ["sound.wav", "sound2.wav", "video.mp4"]


def test_energy_without_frequency_filter(project):
    sampler = EnergyDetectionSampler(project=project, windows_length=100, windows_spacing=100, windows_count=1)
    import numpy as np
    assert sampler.compute_energy_loudness(np.array([1.0, -2.0]), 4096) == 5.0


def test_energy_detection_with_pool(project):
    assert energy(project, threshold=0.98, recordings=AUDIO, ).sample().shape[0] == \
        EnergyDetectionSampler(project=project, windows_length=100, windows_spacing=100, windows_count=1, low_freq=200,
                               high_freq=1000, threads=2, threshold=0.98, recordings=AUDIO).sample().shape[0]


def test_energy_detection_unreadable_recording(project, caplog):
    (project.path / "recordings" / "raw" / "sound2.wav").write_bytes(b"not audio")
    with caplog.at_level(logging.ERROR, logger="ChildProject.pipelines.samplers"):
        assert energy(project).get_recording_windows({"recording_filename": "sound2.wav"}).shape[0] == 0
    assert "failed to read" in caplog.text


def test_energy_detection_recording_missing_from_profile(project, caplog):
    project.write_profile("conv", pd.DataFrame({"original_filename": ["sound.wav"], "converted_filename": ["s.wav"]}))
    with caplog.at_level(logging.ERROR, logger="ChildProject.pipelines.samplers"):
        assert energy(project, profile="conv").get_recording_windows({"recording_filename": "sound2.wav"}).shape[0] == 0
    assert "failed to retrieve the path" in caplog.text


# ============================================================ high volubility
HV_SEGMENTS = pd.DataFrame([
    ("CHI", 0, 200, 0, 1, "NT"), ("FEM", 300, 500, 5, 0, "TIFR"), ("CHI", 600, 800, 0, 1, "TIMR"),
    ("MAL", 2100, 2300, 2, 0, "NT"), ("CHI", 2400, 2600, 0, 3, "TIFR"),
], columns=["speaker_type", "segment_onset", "segment_offset", "words", "utterances_count", "lena_conv_turn_type"])


@pytest.fixture
def voluble(am):
    import_static(am, HV_SEGMENTS.drop(columns=["utterances_count", "lena_conv_turn_type"]), "hv", range_offset=4000)
    import_static(am, HV_SEGMENTS, "hv_lena", range_offset=4000)
    return am.project


def high_volubility(project, metric, annotation_set="hv", count=2, **kwargs):
    kwargs.setdefault("recordings", ["sound.wav"])
    return HighVolubilitySampler(project, annotation_set, metric, windows_length=1000, windows_count=count, **kwargs).sample()


FIRST, SECOND = [0, 800], [2100, 2600]  # windows of the chunks ending in [0, 1000[ and [2000, 3000[


@pytest.mark.parametrize("metric,annotation_set,values,windows", [
    ("turns", "hv", [2, 1], [FIRST, SECOND]),
    ("vocs", "hv", [3, 2], [FIRST, SECOND]),
    ("words", "hv", [5, 2], [FIRST, SECOND]),
    ("turns", "hv_lena", [2, 1], [FIRST, SECOND]),          # lena_conv_turn_type TIFR/TIMR
    ("vocs", "hv_lena", [3, 2], [SECOND, FIRST]),           # sum of utterances_count
])
def test_high_volubility_scores(voluble, metric, annotation_set, values, windows):
    segments = high_volubility(voluble, metric, annotation_set)
    assert segments[metric].tolist() == values
    assert segments[["segment_onset", "segment_offset"]].values.tolist() == windows


def test_high_volubility_windows_count(voluble):
    assert high_volubility(voluble, "words", count=1)["words"].tolist() == [5]


def test_high_volubility_speakers(voluble):
    assert high_volubility(voluble, "vocs", speakers=["CHI"])["vocs"].tolist() == [2, 1]


def test_high_volubility_unknown_metric(voluble):
    with pytest.raises(ValueError, match="unknown metric 'loudness'"):
        high_volubility(voluble, "loudness")


def test_high_volubility_with_pool(voluble):
    assert high_volubility(voluble, "words", threads=2)["words"].tolist() == [5, 2]


def test_high_volubility_recording_without_annotations(voluble):
    segments = high_volubility(voluble, "words", recordings=None)
    assert segments["recording_filename"].unique().tolist() == ["sound.wav"]


# ============================================================ conversations
@pytest.fixture
def conversations(long_am):
    rows = []
    for onset, vocs in [(0, 5), (60_000, 10), (1_800_000, 15)]:
        rows += [{"segment_onset": onset + i * 2500, "segment_offset": onset + i * 2500 + 2000,
                  "speaker_type": ["FEM", "CHI"][i % 2]} for i in range(vocs)]
    import_static(long_am, pd.DataFrame(rows), "conv", range_offset=100_000_000)
    return long_am.project


def test_conversation_sampler_ranks_by_turns(conversations):
    segments = ConversationSampler(conversations, "conv", count=5, interval=1000, speakers=["FEM", "CHI"],
                                   recordings=["sound.wav"]).sample()
    print(segments)
    assert segments["segment_onset"].tolist() == [1_800_000, 60_000, 0]
    assert segments["turns"].tolist() == [14, 9, 4]


def test_conversation_sampler_count(conversations):
    segments = ConversationSampler(conversations, "conv", count=1, recordings=["sound.wav"]).sample()
    assert segments["segment_onset"].tolist() == [1_800_000]


def test_conversation_sampler_interval_merges_blocks(conversations):
    segments = ConversationSampler(conversations, "conv", count=5, interval=10_000_000, recordings=["sound.wav"]).sample()
    assert segments.shape[0] == 1


def test_conversation_sampler_with_pool(conversations):
    assert ConversationSampler(conversations, "conv", count=5, threads=2, recordings=["sound.wav"]).sample().shape[0] == 3


def test_conversation_sampler_recording_without_annotations(conversations, caplog):
    with caplog.at_level(logging.WARNING, logger="ChildProject.pipelines.samplers"):
        segments = ConversationSampler(conversations, "conv", count=5).sample()
    assert "no annotations from the set 'conv' were found for the recording 'sound2.wav'" in caplog.text
    assert segments["recording_filename"].unique().tolist() == ["sound.wav"]


# ============================================================ SamplerPipeline
def test_sampler_pipeline_exports_segments(project, tmp_path):
    path = SamplerPipeline().run(project.path, tmp_path / "out", "periodic", length=1000, period=1000, recordings=AUDIO)
    exported = pd.read_csv(path)
    assert sorted(exported.columns) == ["recording_filename", "segment_offset", "segment_onset"]
    assert exported.shape[0] == 4


def test_sampler_pipeline_exports_parameters(project, tmp_path):
    SamplerPipeline().run(project.path, tmp_path / "out", "periodic", length=1000, period=1000)
    parameters_file = [f for f in os.listdir(tmp_path / "out") if f.startswith("parameters_")]
    assert len(parameters_file) == 1
    content = yaml.unsafe_load(open(tmp_path / "out" / parameters_file[0]))
    assert {"sampler": "periodic"} in content["parameters"]
    assert {"length": 1000} in content["parameters"]


def test_sampler_pipeline_unknown_sampler(project, tmp_path):
    with pytest.raises(NotImplementedError, match="invalid pipeline 'nope'"):
        SamplerPipeline().run(project.path, tmp_path, "nope")
