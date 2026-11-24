"""Zooniverse pipeline (ChildProject.pipelines.zooniverse), run against the fake Panoptes API"""
import logging
import os
import shutil

import pandas as pd
import pytest
import yaml
from pydub import AudioSegment

import ChildProject.pipelines.fake_panoptes as fake
from ChildProject.pipelines.samplers import PeriodicSampler
from ChildProject.pipelines.zooniverse import CHUNKS_DTYPES, Chunk, ZooniversePipeline, pad_interval

from conftest import TEST_DATA, TRUTH

ZOO_TRUTH = TRUTH / "zoochunks"
LOGGER = "ChildProject.pipelines.zooniverse"
CREDENTIALS = dict(zooniverse_login="test_username", zooniverse_pwd="test_password")


@pytest.fixture(autouse=True)
def _reset_fake_api():
    """the fake API keeps global counters; Subject.max_subjects and SubjectSet.ss_n are not restored by reset_tests()"""
    fake.reset_tests()
    fake.SubjectSet.ss_n = 0
    yield
    fake.reset_tests()
    fake.Subject.max_subjects = 10000
    fake.SubjectSet.ss_n = 0


def read_chunks(path):
    return pd.read_csv(path, dtype=CHUNKS_DTYPES)


# ============================================================ helpers
@pytest.mark.parametrize("onset,offset,length,amount,expected", [
    (300, 800, 500, 1, (300, 800)),
    (200, 800, 500, 2, (0, 1000)),
    (300, 900, 500, 1, (100, 1100)),
    (2000, 2500, 100, 10, (1750, 2750)),
    (100, 300, 500, 1, (-50, 450)),
])
def test_pad_interval(onset, offset, length, amount, expected):
    assert pad_interval(onset, offset, length, amount) == expected


def test_chunk_basename():
    assert Chunk("lena/rec.wav", 10, 20, 0, 30).getbasename("mp3") == "lena_rec_10_20.mp3"


def test_get_credentials_explicit():
    assert ZooniversePipeline().get_credentials("me", "pwd") == ("me", "pwd")


def test_get_credentials_from_environment(monkeypatch):
    monkeypatch.setenv("ZOONIVERSE_LOGIN", "env_me")
    monkeypatch.setenv("ZOONIVERSE_PWD", "env_pwd")
    assert ZooniversePipeline().get_credentials() == ("env_me", "env_pwd")


@pytest.mark.parametrize("env,missing", [({}, "login"), ({"ZOONIVERSE_LOGIN": "me"}, "password")])
def test_get_credentials_missing(monkeypatch, env, missing):
    monkeypatch.delenv("ZOONIVERSE_LOGIN", raising=False)
    monkeypatch.delenv("ZOONIVERSE_PWD", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(Exception, match=f"no {missing} specified"):
        ZooniversePipeline().get_credentials()


# ============================================================ extract_chunks
@pytest.fixture
def sampled(project, tmp_path):
    PeriodicSampler(project, 500, 500, 250, recordings=["sound.wav", "sound2.wav"]).sample().to_csv(tmp_path / "segments.csv")
    return project, tmp_path / "segments.csv"


def extract(project, segments, destination, **kwargs):
    return ZooniversePipeline().extract_chunks(path=project.path, destination=str(destination), keyword="test",
                                               segments=str(segments), **kwargs)


def test_extract_chunks_count(sampled, tmp_path):
    project, segments = sampled
    chunks, _ = extract(project, segments, tmp_path / "zoo", chunks_length=250, chunks_min_amount=2)
    assert len(read_chunks(chunks)) == 2 * len(pd.read_csv(segments))


@pytest.mark.parametrize("column", ["wav", "mp3", "png"])
def test_extract_chunks_files(sampled, tmp_path, column):
    project, segments = sampled
    chunks, _ = extract(project, segments, tmp_path / "zoo", chunks_length=250, chunks_min_amount=2, spectrogram=True)
    assert all((tmp_path / "zoo" / "chunks" / f).exists() for f in read_chunks(chunks)[column])


def test_extract_chunks_without_spectrogram(sampled, tmp_path):
    project, segments = sampled
    chunks, _ = extract(project, segments, tmp_path / "zoo", chunks_length=250, chunks_min_amount=2)
    assert read_chunks(chunks)["png"].isna().all()  # written as 'NA'
    assert not list((tmp_path / "zoo" / "chunks").glob("*.png"))


def test_extract_chunks_metadata(sampled, tmp_path):
    project, segments = sampled
    chunks = read_chunks(extract(project, segments, tmp_path / "zoo")[0])
    assert (chunks["keyword"] == "test").all() and (chunks["dataset"] == "test").all()
    assert (chunks["uploaded"] == False).all()


def test_extract_chunks_without_length_keeps_segments(sampled, tmp_path):
    project, segments = sampled
    chunks = read_chunks(extract(project, segments, tmp_path / "zoo")[0])
    assert (chunks["onset"] == chunks["segment_onset"]).all() and (chunks["offset"] == chunks["segment_offset"]).all()
    wav = tmp_path / "zoo" / "chunks" / chunks["wav"].iloc[0]
    assert len(AudioSegment.from_file(wav)) == 500


def test_extract_chunks_copies_segments_and_parameters(sampled, tmp_path):
    project, segments = sampled
    _, parameters = extract(project, segments, tmp_path / "zoo")
    assert (tmp_path / "zoo" / "segments.csv").exists()
    assert {"keyword": "test"} in yaml.unsafe_load(open(parameters))["parameters"]


def test_extract_chunks_skips_negative_onsets(project, tmp_path, caplog):
    pd.DataFrame([["sound.wav", 0, 100]], columns=["recording_filename", "segment_onset", "segment_offset"]).to_csv(tmp_path / "s.csv")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        chunks = read_chunks(extract(project, tmp_path / "s.csv", tmp_path / "zoo", chunks_length=500)[0])
    assert len(chunks) == 0
    assert "skipping chunk with negative onset" in caplog.text


def test_extract_chunks_does_not_overwrite_existing_files(sampled, tmp_path, caplog):
    project, segments = sampled
    extract(project, segments, tmp_path / "zoo", spectrogram=True)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        extract(project, segments, tmp_path / "zoo", spectrogram=True)
    assert "already exists, exportation skipped" in caplog.text


def test_extract_chunks_from_video(project, tmp_path):
    pd.DataFrame([["video.mp4", 1000, 2000]], columns=["recording_filename", "segment_onset", "segment_offset"]).to_csv(tmp_path / "s.csv")
    chunks = read_chunks(extract(project, tmp_path / "s.csv", tmp_path / "zoo")[0])
    assert len(AudioSegment.from_file(tmp_path / "zoo" / "chunks" / chunks["mp3"].iloc[0])) == pytest.approx(1000, abs=30)


def test_extract_chunks_with_pool(sampled, tmp_path):
    project, segments = sampled
    assert len(read_chunks(extract(project, segments, tmp_path / "zoo", threads=2)[0])) == len(pd.read_csv(segments))


def test_extract_chunks_requires_columns(project, tmp_path):
    from ChildProject.tables import MissingColumnsException
    pd.DataFrame({"recording_filename": ["sound.wav"]}).to_csv(tmp_path / "s.csv")
    with pytest.raises(MissingColumnsException):
        extract(project, tmp_path / "s.csv", tmp_path / "zoo")


def test_extract_chunks_requires_segments(project, tmp_path):
    pd.DataFrame(columns=["recording_filename", "segment_onset", "segment_offset"]).to_csv(tmp_path / "s.csv")
    with pytest.raises(AssertionError, match="should not be empty"):
        extract(project, tmp_path / "s.csv", tmp_path / "zoo")


# ============================================================ upload_chunks
@pytest.mark.parametrize("location_fail,amount,ignore_errors,record_orphan,endpoint,result", [
    (False, 1000, False, False, True, "stale_obj_stop.csv"),
    (True, 30, False, False, True, "loc_invalid_stop.csv"),
    (False, 1000, True, True, True, "stale_obj_continue_orphan.csv"),
    (False, 30, True, False, True, "no_errors.csv"),
    (True, 30, True, True, True, "loc_invalid_continue.csv"),
    (True, 30, True, True, 2, "max_subjects_continue.csv"),
])
def test_upload_chunks(tmp_path, location_fail, amount, ignore_errors, record_orphan, endpoint, result):
    chunks = pd.read_csv(TEST_DATA / "chunks_test.csv")
    if not location_fail:
        chunks = chunks[chunks["mp3"] != fake.LOCATION_FAIL]
    path = tmp_path / "chunks.csv"
    chunks.to_csv(path, index=False)
    ZooniversePipeline().upload_chunks(str(path), 404, "test_subject_set", amount=amount, ignore_errors=ignore_errors,
                                       record_orphan=record_orphan, test_endpoint=endpoint, **CREDENTIALS)
    uploaded = read_chunks(path)
    # subject_set_id in the truth files came from the fake API's global counter in the old test order
    pd.testing.assert_frame_equal(read_chunks(ZOO_TRUTH / result).drop(columns="subject_set_id"),
                                  uploaded.drop(columns="subject_set_id"), check_like=True)
    assert set(uploaded.loc[uploaded["subject_set"] == "test_subject_set", "subject_set_id"].dropna().astype(int)) <= {2}


def test_upload_chunks_into_existing_subject_set(tmp_path):
    path = tmp_path / "chunks.csv"
    pd.read_csv(TEST_DATA / "chunks_test.csv").head(1).to_csv(path, index=False)
    ZooniversePipeline().upload_chunks(str(path), 404, "ss1", test_endpoint=True, **CREDENTIALS)
    assert read_chunks(path)["subject_set"].tolist() == ["ss1"]


def test_upload_chunks_nothing_left(tmp_path, caplog):
    path = tmp_path / "chunks.csv"
    pd.read_csv(TEST_DATA / "chunks_test.csv").assign(uploaded=True).to_csv(path, index=False)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert ZooniversePipeline().upload_chunks(str(path), 404, "s", test_endpoint=True, **CREDENTIALS) is None
    assert "nothing left to upload" in caplog.text


def test_upload_chunks_unreadable_metadata(tmp_path):
    with pytest.raises(Exception, match="cannot read chunk metadata"):
        ZooniversePipeline().upload_chunks(str(tmp_path / "missing.csv"), 404, "s", test_endpoint=True, **CREDENTIALS)


def test_upload_chunks_requires_columns(tmp_path):
    from ChildProject.tables import MissingColumnsException
    path = tmp_path / "chunks.csv"
    pd.read_csv(TEST_DATA / "chunks_test.csv").drop(columns=["mp3"]).to_csv(path, index=False)
    with pytest.raises(MissingColumnsException):
        ZooniversePipeline().upload_chunks(str(path), 404, "s", test_endpoint=True, **CREDENTIALS)


def test_upload_chunks_restores_signal_handlers(tmp_path):
    import signal
    before = signal.getsignal(signal.SIGINT)
    path = tmp_path / "chunks.csv"
    pd.read_csv(TEST_DATA / "chunks_test.csv").head(2).to_csv(path, index=False)
    ZooniversePipeline().upload_chunks(str(path), 404, "s", test_endpoint=True, **CREDENTIALS)
    assert signal.getsignal(signal.SIGINT) is before


def test_exit_upload_saves_progress(tmp_path):
    pipeline = ZooniversePipeline()
    pipeline.chunks_file = str(tmp_path / "chunks.csv")
    pipeline.chunks = pd.read_csv(TEST_DATA / "chunks_test.csv", index_col="index", dtype=CHUNKS_DTYPES).head(2)
    pipeline.subjects_metadata = [{"index": 0, "uploaded": True, "zooniverse_id": "7"}]
    pipeline.orphan_chunks = [{"index": 1, "uploaded": True, "zooniverse_id": "8"}]
    with pytest.raises(SystemExit):
        pipeline.exit_upload(2, None, rec_orphan=True, sub_set="s")
    saved = read_chunks(tmp_path / "chunks.csv")
    assert saved["zooniverse_id"].astype(str).tolist() == ["7", "8"]


# ============================================================ orphan subjects
@pytest.fixture
def orphans(tmp_path):
    path = tmp_path / "orphans.csv"
    shutil.copy(TEST_DATA / "chunks_test_orphan.csv", path)
    return path


@pytest.mark.parametrize("ignore_errors,result", [(False, "link_orphan_stop.csv"), (True, "link_orphan_continue.csv")])
def test_link_orphan_subjects(orphans, ignore_errors, result):
    ZooniversePipeline().link_orphan_subjects(str(orphans), 404, "test_subject_set", ignore_errors=ignore_errors,
                                              test_endpoint=True, **CREDENTIALS)
    linked = read_chunks(orphans)
    # truth files hold a subject_set_id that depended on the fake API's global counter in the old test order
    pd.testing.assert_frame_equal(read_chunks(ZOO_TRUTH / result).drop(columns="subject_set_id"),
                                  linked.drop(columns="subject_set_id"))
    # the fake project owns sets 0 and 1, 'test_subject_set' is created with id 2
    assert set(linked.loc[linked["subject_set"] == "test_subject_set", "subject_set_id"].dropna().astype(int)) <= {2}


def test_reset_orphan_subjects(orphans):
    ZooniversePipeline().reset_orphan_subjects(str(orphans))
    pd.testing.assert_frame_equal(read_chunks(ZOO_TRUTH / "reset_orphan.csv"), read_chunks(orphans))


def test_reset_orphan_subjects_logs_count(orphans, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        ZooniversePipeline().reset_orphan_subjects(str(orphans))
    assert any(m.startswith("reset ") and m.endswith(" orphan subjects") for m in caplog.messages)


def test_reset_without_orphans(tmp_path, caplog):
    path = tmp_path / "chunks.csv"
    shutil.copy(ZOO_TRUTH / "reset_orphan.csv", path)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        ZooniversePipeline().reset_orphan_subjects(str(path))
    assert "no orphan subject to reset" in caplog.text


def test_reset_orphan_unreadable_metadata(tmp_path):
    with pytest.raises(Exception, match="cannot read chunk metadata"):
        ZooniversePipeline().reset_orphan_subjects(str(tmp_path / "missing.csv"))


# ============================================================ classifications
class _Workflow:
    id = "wf"
    tasks = {"T0": {"answers": [{"label": "yes"}, {"label": "no"}]}}


class _Project:
    def __init__(self, ident):
        self.links = fake.Links().add_attr("workflows", [_Workflow()])


class _Classification:
    def __init__(self, raw):
        self.raw = raw

    @staticmethod
    def where(scope, page_size, project_id):
        return [_Classification({"id": i, "links": {"user": "u", "subjects": [str(subject)], "workflow": "wf"},
                                 "annotations": [{"task": "T0", "value": answer}]})
                for i, subject, answer in [(1, 7, 0), (2, 8, 1)]]


@pytest.fixture
def classification_api(monkeypatch):
    monkeypatch.setattr(fake, "Project", _Project)
    monkeypatch.setattr(fake, "Classification", _Classification)


def test_retrieve_classifications(classification_api, tmp_path):
    result = ZooniversePipeline().retrieve_classifications(str(tmp_path / "c.csv"), 404, test_endpoint=True, **CREDENTIALS)
    assert result[["subject_id", "answer"]].values.tolist() == [[7, "yes"], [8, "no"]]
    assert pd.read_csv(tmp_path / "c.csv").shape[0] == 2


def test_retrieve_classifications_matched_to_chunks(classification_api, tmp_path):
    chunks = tmp_path / "chunks.csv"
    pd.DataFrame({"zooniverse_id": [8], "recording_filename": ["sound.wav"]}).to_csv(chunks, index=False)
    result = ZooniversePipeline().retrieve_classifications(str(tmp_path / "c.csv"), 404, chunks=[str(chunks)],
                                                           test_endpoint=True, **CREDENTIALS)
    assert result[["subject_id", "recording_filename", "answer"]].values.tolist() == [[8, "sound.wav", "no"]]


def test_retrieve_classifications_with_bundled_fake_api(tmp_path):
    """the fake Panoptes module shipped for tests should support retrieve_classifications"""
    result = ZooniversePipeline().retrieve_classifications(str(tmp_path / "c.csv"), 404, test_endpoint=True, **CREDENTIALS)
    assert result.shape[0] == 1


# ============================================================ run dispatch
@pytest.mark.parametrize("action,method", [
    ("extract-chunks", "extract_chunks"), ("upload-chunks", "upload_chunks"),
    ("link-orphan-subjects", "link_orphan_subjects"), ("reset-orphan-subjects", "reset_orphan_subjects"),
    ("retrieve-classifications", "retrieve_classifications"),
])
def test_run_dispatches_actions(monkeypatch, action, method):
    pipeline = ZooniversePipeline()
    monkeypatch.setattr(pipeline, method, lambda **kwargs: (method, kwargs))
    assert pipeline.run(action, a=1) == (method, {"a": 1})


def test_setup_parser_actions():
    import argparse
    parser = argparse.ArgumentParser()
    ZooniversePipeline.setup_parser(parser)
    args = parser.parse_args(["reset-orphan-subjects", "--chunks", "c.csv"])
    assert (args.action, args.chunks) == ("reset-orphan-subjects", "c.csv")
