import pandas as pd
from pathlib import Path

from ChildProject.projects import ChildProject
import ChildProject.pipelines.derivations as deriv

CP_PATH = Path('examples','valid_raw_data')
CSV_DF = pd.read_csv(Path('examples', 'valid_raw_data', 'annotations', 'vtc_present', 'converted',
                          'sound2_0_4000.csv'))
TRUTH = Path('tests', 'truth', 'derivations')


def test_conversations():
    df = CSV_DF.copy()
    meta = {}
    project = None

    res = deriv.conversations(project, meta, df)
    # res.to_csv(TRUTH / 'conversations.csv', index=False)
    truth = pd.read_csv(TRUTH / 'conversations.csv')

    pd.testing.assert_frame_equal(res, truth, check_dtype=False)

def test_acoustics():
    df = CSV_DF.copy()
    project = ChildProject(CP_PATH)
    project.read()
    meta = {'recording_filename': 'sound.wav'}

    res = deriv.acoustics(project, meta, df, profile=None, target_sr=4096)
    # res.to_csv(TRUTH / 'acoustics.csv', index=False)
    truth = pd.read_csv(TRUTH / 'acoustics.csv')

    print(truth.to_string())
    print(res.to_string())

    pd.testing.assert_frame_equal(res, truth)


def test_remove_overlaps():
    df = CSV_DF.copy()
    meta = {}
    project = None

    res = deriv.remove_overlaps(project, meta, df)
    # res.to_csv(TRUTH / 'remove-overlaps.csv', index=False)
    truth = pd.read_csv(TRUTH / 'remove-overlaps.csv')

    pd.testing.assert_frame_equal(res, truth, check_dtype=False)
