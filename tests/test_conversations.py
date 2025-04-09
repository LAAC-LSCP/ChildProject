from functools import partial
import numpy as np
import os
import pandas as pd
import pytest
import shutil
from pathlib import Path

from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from ChildProject.pipelines.conversations import (Conversations, StandardConversations, CustomConversations,
                                                 ConversationsSpecificationPipeline)

from ChildProject.pipelines.conversationFunctions import conversationFunction, RESERVED

PATH = Path('output/conversations')


def fake_vocs(data, filename):
    return data


@pytest.fixture(scope="function")
def project(request):
    if os.path.exists(PATH):
        # shutil.copytree(src="examples/valid_raw_data", dst="output/annotations")
        shutil.rmtree(PATH)
    shutil.copytree(src="examples/valid_raw_data", dst=PATH, symlinks=True)

    project = ChildProject(PATH)
    project.read()

    yield project


@pytest.fixture(scope="function")
def am(request, project):
    am = AnnotationManager(project)
    project.recordings['duration'] = [100000000, 2000000] #force longer durations to allow for imports
    yield am

@pytest.fixture(scope="function")
def segments(request):
    segments = pd.read_csv("tests/data/csv.csv")
    segments.loc[2:4, 'conv_count'] = 1
    segments.loc[8:9, 'conv_count'] = 2
    segments.loc[10:11, 'conv_count'] = 3

    yield segments


def test_failures(project):
    features = pd.DataFrame([["who_initiated", "initiator", pd.NA],
                             ["who_finished", "finisher", pd.NA],
                             ["voc_speaker_count", "CHI_voc_count", 'CHI'],
                             ], columns=['callable', 'name', 'speaker'])

    exception_caught = False
    try:
        standard = StandardConversations(project, setname="unknown")
    except ValueError as e:
        exception_caught = True

    assert (
            exception_caught is True
    ), "StandardConversations failed to throw an exception despite an invalid set being provided"

    exception_caught = False
    try:
        custom = CustomConversations(project, setname="unknown", features='tests/data/list_features_conv.csv')
    except ValueError as e:
        exception_caught = True

    assert (
            exception_caught is True
    ), "CustomConversations failed to throw an exception despite an invalid set being provided"


@pytest.mark.parametrize("error,col_change,new_value",
                         [(ValueError, 'name', 'finisher'),
                          (ValueError, 'callable', 'made_up_function'),
                          (TypeError, 'speaker', 'FEM'),
                          (None, None, None),
                          ])
def test_conversations(project, am, segments, error, col_change, new_value):

    am.import_annotations(
        pd.DataFrame(
            [{  "set": "custom_conv",
                "raw_filename": "file.its",
                "time_seek": 0,
                "recording_filename": "sound.wav",
                "range_onset": 0,
                "range_offset": 30000000,
                "format": "csv",
            }]
        ),
        import_function=partial(fake_vocs, segments),
    )

    features = pd.DataFrame([["who_initiated", "initiator", pd.NA],
                               ["who_finished", "finisher", pd.NA],
                               ["voc_speaker_count", "CHI_voc_count", 'CHI'],
                              ], columns=['callable', 'name', 'speaker'])

    if error:
        with pytest.raises(error):
            features.iloc[0, features.columns.get_loc(col_change)] = new_value
            cm = Conversations(project, 'custom_conv', features)
            cm.extract()
    else:
        cm = Conversations(project, 'custom_conv', features)
        results = cm.extract()

        # cm.conversations.to_csv("tests/truth/python_conversations.csv",index=False)
        truth = pd.read_csv("tests/truth/python_conversations.csv")
        print(truth['interval_last_conv'])
        print(results['interval_last_conv'])

        pd.testing.assert_frame_equal(results, truth, check_like=True, check_dtype=False)

#TODO adapt
def test_standard(project, am, segments):
    am.import_annotations(
        pd.DataFrame(
            [{"set": "custom_conv",
              "raw_filename": "file.its",
              "time_seek": 0,
              "recording_filename": "sound.wav",
              "range_onset": 0,
              "range_offset": 30000000,
              "format": "csv",
              }]
        ),
        import_function=partial(fake_vocs, segments),
    )

    std = StandardConversations(project, setname='custom_conv', rec_cols='date_iso', child_cols='experiment,child_dob',
                                set_cols='method,has_speaker_type')
    std.extract()

    # std.conversations.to_csv("tests/truth/standard_conversations.csv", index=False)
    truth = pd.read_csv("tests/truth/standard_conversations.csv", dtype={'child_id': str})

    pd.testing.assert_frame_equal(std.conversations, truth, check_like=True, check_dtype=False)


#TODO adapt
def test_custom(project, am, segments):
    am.import_annotations(
        pd.DataFrame(
            [{"set": "custom_conv",
              "raw_filename": "file.its",
              "time_seek": 0,
              "recording_filename": "sound.wav",
              "range_onset": 0,
              "range_offset": 30000000,
              "format": "csv",
              }]
        ),
        import_function=partial(fake_vocs, segments),
    )

    parameters = "tests/data/list_features_conv.csv"

    cm = CustomConversations(project, 'custom_conv', parameters)
    cm.extract()

    # cm.conversations.to_csv("tests/truth/custom_conversations.csv", index=False)
    truth = pd.read_csv("tests/truth/custom_conversations.csv")

    pd.testing.assert_frame_equal(cm.conversations, truth, check_like=True, check_dtype=False)


#TODO adapt
def test_specs(project, am, segments):
    am.import_annotations(
        pd.DataFrame(
            [{"set": "custom_conv",
              "raw_filename": "file.its",
              "time_seek": 0,
              "recording_filename": "sound.wav",
              "range_onset": 0,
              "range_offset": 30000000,
              "format": "csv",
              }]
        ),
        import_function=partial(fake_vocs, segments),
    )

    csp = ConversationsSpecificationPipeline()

    parameters = "tests/data/conversations_parameters.yml"
    csp.run(parameters)

    output = pd.read_csv(csp.destination)
    output.to_csv("tests/truth/specs_conversations.csv", index=False)
    truth = pd.read_csv("tests/truth/specs_conversations.csv")

    pd.testing.assert_frame_equal(output, truth, check_like=True)

    new_params = csp.parameters_path
    csp.run(new_params)

    output = pd.read_csv(csp.destination)

    pd.testing.assert_frame_equal(output, truth, check_like=True)


def test_empty_conversations(project, am):
    empty_segments = pd.DataFrame(columns=["segment_onset", "segment_offset", "speaker_type", "time_since_last_conv", "conv_count"])

    am.import_annotations(
        pd.DataFrame(
            [{"set": "empty_conv",
              "raw_filename": "file.its",
              "time_seek": 0,
              "recording_filename": "sound.wav",
              "range_onset": 0,
              "range_offset": 30000000,
              "format": "csv",
              }]
        ),
        import_function=partial(fake_vocs, empty_segments),
    )

    std = StandardConversations(project, setname='empty_conv')
    results = std.extract()

    assert results.empty, "The result should be empty for an empty dataset"


def test_single_entry_conversation(project, am):
    single_segment = pd.DataFrame({
        "segment_onset": [0],
        "segment_offset": [5],
        "speaker_type": ["CHI"],
        "time_since_last_conv": [np.nan],
        "conv_count": [1]
    })

    am.import_annotations(
        pd.DataFrame(
            [{"set": "single_conv",
              "raw_filename": "file.its",
              "time_seek": 0,
              "recording_filename": "sound.wav",
              "range_onset": 0,
              "range_offset": 30000000,
              "format": "csv",
              }]
        ),
        import_function=partial(fake_vocs, single_segment),
    )

    std = StandardConversations(project, setname='single_conv')
    results = std.extract()

    assert len(results) == 1, "The result should contain one conversation for a single entry dataset"


def test_unsorted_annotations(project, am):
    unsorted_segments = pd.DataFrame({
        "segment_onset": [20, 0, 10],
        "segment_offset": [25, 5, 15],
        "speaker_type": ["FEM", "CHI", "MAN"],
        "time_since_last_conv": [5, np.nan, 15],
        "conv_count": [2, 1, 1]
    })

    am.import_annotations(
        pd.DataFrame(
            [{"set": "unsorted_conv",
              "raw_filename": "file.its",
              "time_seek": 0,
              "recording_filename": "sound.wav",
              "range_onset": 0,
              "range_offset": 30000000,
              "format": "csv",
              }]
        ),
        import_function=partial(fake_vocs, unsorted_segments),
    )

    std = StandardConversations(project, setname='unsorted_conv')
    results = std.extract()

    assert not results.empty, "The result should not be empty for unsorted annotations"
