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
    project = None
    meta = {}

    derivator=deriv.ConversationDerivator()
    res = derivator.derive(project, meta, df)
    # res.to_csv(TRUTH / 'conversations.csv', index=False)
    truth = pd.read_csv(TRUTH / 'conversations.csv')

    pd.testing.assert_frame_equal(res, truth, check_dtype=False)

def test_acoustics():
    df = CSV_DF.copy()
    project = ChildProject(CP_PATH)
    project.read()
    meta = {'recording_filename': 'sound.wav'}

    derivator = deriv.AcousticDerivator(profile=None, target_sr=4096)
    res = derivator.derive(project, meta, df)
    # res.to_csv(TRUTH / 'acoustics.csv', index=False)
    truth = pd.read_csv(TRUTH / 'acoustics.csv')

    pd.testing.assert_frame_equal(res, truth, check_exact=False, rtol=1e-4, atol=1e-7)


def test_remove_overlaps():
    df = CSV_DF.copy()
    project = None
    meta = {}

    derivator = deriv.RemoveOverlapsDerivator()
    res = derivator.derive(project, meta, df)
    # res.to_csv(TRUTH / 'remove-overlaps.csv', index=False)
    truth = pd.read_csv(TRUTH / 'remove-overlaps.csv')

    pd.testing.assert_frame_equal(res, truth, check_dtype=False)

def test_cva():
    df = CSV_DF.copy()
    meta = {}
    project = None

    # we use restrictive, so we should remove overlaps first (this is not ideal for independent testing though)
    derivator = deriv.RemoveOverlapsDerivator()
    df = derivator.derive(project, meta, df)
    derivator = deriv.CVADerivator()
    res = derivator.derive(project, meta, df)
    # res.to_csv(TRUTH / 'cva.csv', index=False)
    truth = pd.read_csv(TRUTH / 'cva.csv', keep_default_na=False)

    pd.testing.assert_frame_equal(res, truth, check_dtype=False)
