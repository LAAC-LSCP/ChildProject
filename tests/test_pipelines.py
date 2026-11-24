"""ChildProject.pipelines.pipeline: base class and recordings selection"""
import pandas as pd
import pytest

from ChildProject.pipelines.pipeline import Pipeline


class MinimalPipeline(Pipeline):
    def run(self, **kwargs):
        return kwargs


def test_pipeline_is_abstract():
    with pytest.raises(TypeError):
        Pipeline()


def test_pipeline_default_hooks_do_nothing():
    pipeline = MinimalPipeline()
    assert pipeline.check_setup() is None
    assert pipeline.setup() is None
    assert Pipeline.setup_pipeline(None) is None
    assert pipeline.run(a=1) == {"a": 1}


def test_recordings_from_list_none():
    assert Pipeline.recordings_from_list(None) is None


@pytest.mark.parametrize("recordings", [
    ["sound.wav", "sound2.wav"],
    pd.Series(["sound.wav", "sound2.wav"]),
    pd.DataFrame({"recording_filename": ["sound.wav", "sound2.wav"]}),
])
def test_recordings_from_list(recordings):
    assert sorted(Pipeline.recordings_from_list(recordings)) == ["sound.wav", "sound2.wav"]


def test_recordings_from_list_removes_duplicates():
    assert Pipeline.recordings_from_list(["sound.wav", "sound.wav"]) == ["sound.wav"]


def test_recordings_from_list_csv(tmp_path):
    path = tmp_path / "filter.csv"
    pd.DataFrame({"recording_filename": ["sound.wav"]}).to_csv(path)
    assert Pipeline.recordings_from_list(str(path)) == ["sound.wav"]


def test_recordings_from_list_dataframe_without_column():
    with pytest.raises(ValueError, match="missing a 'recording_filename' column"):
        Pipeline.recordings_from_list(pd.DataFrame({"filename": ["sound.wav"]}))


def test_recordings_from_list_csv_without_column(tmp_path):
    path = tmp_path / "filter.csv"
    pd.DataFrame({"filename": ["sound.wav"]}).to_csv(path)
    with pytest.raises(ValueError, match="is missing a 'recording_filename' column"):
        Pipeline.recordings_from_list(str(path))


def test_recordings_from_list_missing_file(tmp_path):
    with pytest.raises(ValueError, match="is neither a pandas dataframe"):
        Pipeline.recordings_from_list(str(tmp_path / "nope.csv"))


def test_recordings_from_list_used_with_project(project):
    names = Pipeline.recordings_from_list([str(project.path / "recordings/raw/sound.wav")])
    assert project.get_recordings_from_list(names)["recording_filename"].tolist() == ["sound.wav"]
