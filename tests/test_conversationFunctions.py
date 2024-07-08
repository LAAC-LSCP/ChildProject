import pandas as pd
import pytest

import ChildProject.pipelines.conversationFunctions as cf

@pytest.fixture(scope="function")
def segments(request):
    segments = pd.read_csv("tests/data/csv.csv").dropna(subset=['speaker_type'])
    segments['voc_duration'] = segments['segment_offset'] - segments['segment_onset']

    yield segments

@pytest.mark.parametrize("function,parameters,truth",
                         [(cf.who_initiated, {}, 'CHI'),
                          (cf.who_finished, {}, 'MAL'),
                          (cf.participants, {}, 'CHI/OCH/FEM/MAL'),
                          (cf.voc_total_dur, {}, 15034),
                          (cf.is_speaker, {'speaker': 'XXX'}, False),
                          (cf.is_speaker, {'speaker': 'OCH'}, True),
                          (cf.voc_speaker_count, {'speaker': 'CHI'}, 1),
                          (cf.voc_speaker_count, {'speaker': 'OCH'}, 3),
                          (cf.voc_speaker_dur, {'speaker': 'MAL'}, 3924),
                          (cf.voc_speaker_dur, {'speaker': 'FEM'}, 3459),
                          (cf.voc_dur_contribution, {'speaker': 'FEM'}, 3459/15034),
                          (cf.voc_dur_contribution, {'speaker': 'OCH'}, 6794/15034),
                          (cf.assign_conv_type, {}, 'multiparty'),
                          ])
def test_conversations(segments, function, parameters, truth):
    result = function(segments, **parameters)

    assert result == truth