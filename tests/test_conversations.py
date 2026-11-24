"""conversation extraction pipelines (ChildProject.pipelines.conversations)"""
import logging

import numpy as np
import pandas as pd
import pytest
import yaml

from ChildProject.pipelines.conversations import (
    Conversations,
    ConversationsPipeline,
    ConversationsSpecificationPipeline,
    CustomConversations,
    StandardConversations,
    pipelines,
)

from conftest import TEST_DATA, TRUTH, import_static

LOGGER = "ChildProject.pipelines.conversations"


def features(rows=(("who_initiated", "initiator", pd.NA), ("who_finished", "finisher", pd.NA),
                   ("voc_speaker_count", "CHI_voc_count", "CHI"))):
    return pd.DataFrame(list(rows), columns=["callable", "name", "speaker"])


@pytest.fixture
def conv_project(long_am, conversation_segments):
    """dataset with set 'custom_conv' holding 3 labelled conversations for sound.wav"""
    import_static(long_am, conversation_segments, "custom_conv")
    return long_am.project


def test_pipelines_registry():
    assert pipelines == {"custom": CustomConversations, "standard": StandardConversations}


# ============================================================ input checks
def test_unknown_set(conv_project):
    with pytest.raises(ValueError, match="annotation set 'unknown' was not found"):
        Conversations(conv_project, "unknown", features())


@pytest.mark.parametrize("cls,kwargs", [(StandardConversations, {}), (CustomConversations, {"features": "tests/data/list_features_conv.csv"})])
def test_subclasses_check_set(conv_project, cls, kwargs):
    with pytest.raises(ValueError, match="was not found in the index"):
        cls(conv_project, setname="unknown", **kwargs)


def test_unknown_feature(conv_project):
    with pytest.raises(ValueError, match="made_up_function function is not defined"):
        Conversations(conv_project, "custom_conv", features([("made_up_function", "x", pd.NA)]))


def test_feature_must_be_callable_or_name(conv_project):
    with pytest.raises(ValueError, match="cannot be evaluated as a feature"):
        Conversations(conv_project, "custom_conv", features([(3, "x", pd.NA)]))


def test_duplicated_feature_names(conv_project):
    with pytest.raises(ValueError, match="duplicates in 'name' column"):
        Conversations(conv_project, "custom_conv", features([("who_initiated", "x", pd.NA), ("who_finished", "x", pd.NA)]))


def test_features_require_columns(conv_project):
    with pytest.raises(ValueError, match=r"must contain at least the columns \[callable,name\]"):
        Conversations(conv_project, "custom_conv", pd.DataFrame({"callable": ["who_initiated"]}))


def test_features_must_be_dataframe(conv_project):
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        Conversations(conv_project, "custom_conv", "tests/data/list_features_conv.csv")


def test_feature_missing_argument_fails_at_extraction(conv_project):
    conversations = Conversations(conv_project, "custom_conv", features([("voc_speaker_count", "count", pd.NA)]))
    with pytest.raises(ValueError, match="needs an argument <speaker>"):
        conversations.extract()


@pytest.mark.parametrize("option,value", [("from_time", "10:00"), ("to_time", "10h")])
def test_invalid_time(conv_project, option, value):
    with pytest.raises(ValueError, match=f"invalid value for {option}"):
        Conversations(conv_project, "custom_conv", features(), **{option: value})


def test_unknown_requested_columns_are_ignored(conv_project, caplog, capsys):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = Conversations(conv_project, "custom_conv", features(), rec_cols="nope", child_cols="nope",
                               set_cols="nope").extract()
    assert "requested column <nope> does not exist in recordings.csv" in caplog.text
    assert "does not exist in children.csv" in caplog.text
    assert "does not exist in the set metadata" in capsys.readouterr().out
    assert "nope" not in result.columns


# ============================================================ extraction
def test_extract_matches_truth(conv_project):
    result = Conversations(conv_project, "custom_conv", features()).extract()
    pd.testing.assert_frame_equal(result, pd.read_csv(TRUTH / "python_conversations.csv"), check_like=True, check_dtype=False)


def test_extract_one_row_per_conversation(conv_project):
    result = Conversations(conv_project, "custom_conv", features()).extract()
    assert result["conv_count"].tolist() == [1, 2, 3]
    assert (result["recording_filename"] == "sound.wav").all()


def test_extract_interval_since_previous_conversation(conv_project, conversation_segments):
    result = Conversations(conv_project, "custom_conv", features()).extract()
    s = conversation_segments
    assert pd.isna(result.loc[0, "interval_last_conv"])
    assert result.loc[1, "interval_last_conv"] == s.loc[8, "segment_onset"] - s.loc[4, "segment_offset"]


def test_extract_with_pool(conv_project):
    result = Conversations(conv_project, "custom_conv", features(), threads=2).extract()
    pd.testing.assert_frame_equal(result, pd.read_csv(TRUTH / "python_conversations.csv"), check_like=True, check_dtype=False)


def test_extract_restricted_recordings(conv_project):
    result = Conversations(conv_project, "custom_conv", features(), recordings=["sound2.wav"]).extract()
    assert result.shape[0] == 0


@pytest.mark.parametrize("start,end,expected", [
    # sound.wav starts at 9:00; conversation 1 is ~33 min in (09:33), conversations 2-3 ~7h51 in (16:51)
    ("09:30:00", "10:00:00", [1]),
    ("16:30:00", "17:00:00", [2, 3]),
    ("12:00:00", "13:00:00", []),
])
def test_extract_time_range(conv_project, start, end, expected):
    result = Conversations(conv_project, "custom_conv", features(), from_time=start, to_time=end).extract()
    assert result["conv_count"].tolist() == expected


@pytest.mark.parametrize("kwargs,columns", [
    ({"rec_cols": "date_iso"}, {"date_iso"}),
    ({"child_cols": "child_dob"}, {"child_dob", "child_id"}),
    ({"rec_cols": "date_iso", "child_cols": "child_dob"}, {"date_iso", "child_dob"}),
    ({"set_cols": "method"}, {"method"}),
])
def test_extract_metadata_columns(conv_project, kwargs, columns):
    result = Conversations(conv_project, "custom_conv", features(), **kwargs).extract()
    assert columns <= set(result.columns)


def test_extract_set_column_already_in_results(conv_project, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        Conversations(conv_project, "custom_conv", features([("who_initiated", "method", pd.NA)]), set_cols="method").extract()
    assert "Ignoring required column method" in caplog.text


def test_standard_conversations(conv_project):
    result = StandardConversations(conv_project, setname="custom_conv", rec_cols="date_iso",
                                   child_cols="experiment,child_dob", set_cols="method,has_speaker_type").extract()
    truth = pd.read_csv(TRUTH / "standard_conversations.csv", dtype={"child_id": str})
    pd.testing.assert_frame_equal(result, truth, check_like=True, check_dtype=False)


def test_custom_conversations(conv_project):
    result = CustomConversations(conv_project, "custom_conv", TEST_DATA / "list_features_conv.csv").extract()
    pd.testing.assert_frame_equal(result, pd.read_csv(TRUTH / "custom_conversations.csv"), check_like=True, check_dtype=False)


def _import(long_am, segments, name):
    import_static(long_am, pd.DataFrame(segments), name)
    return long_am.project


def test_empty_set_gives_no_conversation(long_am, caplog):
    project = _import(long_am, {c: [] for c in ["segment_onset", "segment_offset", "speaker_type", "conv_count"]}, "empty_conv")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert StandardConversations(project, setname="empty_conv").extract().empty
    assert "did not find any conversation" in caplog.text


def test_single_vocalization_conversation(long_am):
    project = _import(long_am, {"segment_onset": [0], "segment_offset": [5], "speaker_type": ["CHI"], "conv_count": [1]}, "single")
    assert len(StandardConversations(project, setname="single").extract()) == 1


def test_unsorted_annotations(long_am):
    project = _import(long_am, {"segment_onset": [20, 0, 10], "segment_offset": [25, 5, 15],
                                "speaker_type": ["FEM", "CHI", "MAL"], "conv_count": [2, 1, 1]}, "unsorted")
    result = StandardConversations(project, setname="unsorted").extract()
    assert result["conv_count"].tolist() == [1, 2]
    assert result["initiator"].tolist() == ["CHI", "FEM"]


def test_vocalizations_outside_conversations_are_ignored(long_am):
    project = _import(long_am, {"segment_onset": [0, 10, 20], "segment_offset": [5, 15, 25],
                                "speaker_type": ["CHI", "FEM", "MAL"], "conv_count": [1, np.nan, 1]}, "gaps")
    assert StandardConversations(project, setname="gaps").extract().loc[0, "voc_count"] == 2


# ============================================================ ConversationsPipeline
def test_conversations_pipeline_writes_results(conv_project, tmp_path):
    destination = tmp_path / "conv.csv"
    pipeline = ConversationsPipeline()
    pipeline.run(str(conv_project.path), str(destination), "standard", setname="custom_conv")
    assert pd.read_csv(destination).shape[0] == 3


def test_conversations_pipeline_writes_parameters(conv_project, tmp_path, caplog):
    pipeline = ConversationsPipeline()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        pipeline.run(str(conv_project.path), str(tmp_path / "conv.csv"), "standard", setname="custom_conv")
    parameters = yaml.safe_load(open(pipeline.parameters_path))["parameters"]
    assert parameters["pipeline"] == "standard"
    assert {"callable": "who_initiated", "name": "initiator"} in parameters["features_list"]
    assert "not currently a git repository" in caplog.text


def test_conversations_pipeline_unknown_pipeline(conv_project, tmp_path):
    with pytest.raises(NotImplementedError, match="invalid pipeline 'nope'"):
        ConversationsPipeline().run(str(conv_project.path), str(tmp_path / "c.csv"), "nope", setname="custom_conv")


# ============================================================ ConversationsSpecificationPipeline
def write_spec(tmp_path, project, **overrides):
    spec = yaml.safe_load((TEST_DATA / "conversations_parameters.yml").read_text())
    spec["parameters"].update({"path": str(project.path), "destination": str(tmp_path / "std_conv.csv")})
    spec["parameters"].update(overrides)
    path = tmp_path / "spec.yml"
    path.write_text(yaml.dump(spec))
    return path


def test_specification_pipeline(conv_project, tmp_path):
    pipeline = ConversationsSpecificationPipeline()
    pipeline.run(write_spec(tmp_path, conv_project))
    pd.testing.assert_frame_equal(pd.read_csv(pipeline.destination), pd.read_csv(TRUTH / "specs_conversations.csv"),
                                  check_like=True)


def test_specification_pipeline_matches_standard_pipeline(conv_project, tmp_path):
    spec = ConversationsSpecificationPipeline()
    spec.run(write_spec(tmp_path, conv_project))
    standard = StandardConversations(conv_project, setname="custom_conv").extract()
    pd.testing.assert_frame_equal(pd.read_csv(spec.destination), standard.astype(object).infer_objects(),
                                  check_like=True, check_dtype=False)


def test_specification_pipeline_rerun_from_exported_parameters(conv_project, tmp_path):
    pipeline = ConversationsSpecificationPipeline()
    pipeline.run(write_spec(tmp_path, conv_project))
    first = pd.read_csv(pipeline.destination)
    pipeline.run(pipeline.parameters_path)
    pd.testing.assert_frame_equal(pd.read_csv(pipeline.destination), first, check_like=True)


@pytest.mark.parametrize("missing", ["path", "destination", "features_list"])
def test_specification_pipeline_required_keys(conv_project, tmp_path, missing):
    spec = write_spec(tmp_path, conv_project)
    content = yaml.safe_load(spec.read_text())
    del content["parameters"][missing]
    spec.write_text(yaml.dump(content))
    with pytest.raises(ValueError, match=f"'{missing}' key"):
        ConversationsSpecificationPipeline().run(spec)


def test_specification_pipeline_empty_file(tmp_path):
    (tmp_path / "empty.yml").write_text("")
    with pytest.raises(ValueError, match="could not find any parameters"):
        ConversationsSpecificationPipeline().run(tmp_path / "empty.yml")


def test_specification_pipeline_invalid_yaml(tmp_path):
    (tmp_path / "bad.yml").write_text("a: [b\n")
    with pytest.raises(yaml.YAMLError, match="parsing of the parameters file"):
        ConversationsSpecificationPipeline().run(tmp_path / "bad.yml")


def test_specification_pipeline_features_list_must_be_a_list(conv_project, tmp_path):
    with pytest.raises(ValueError, match="must be a list of elements"):
        ConversationsSpecificationPipeline().run(write_spec(tmp_path, conv_project, features_list=5))


def test_specification_pipeline_unknown_parameter(conv_project, tmp_path):
    with pytest.raises(ValueError, match="Unrecognized parameter found"):
        ConversationsSpecificationPipeline().run(write_spec(tmp_path, conv_project, colour="blue"))
