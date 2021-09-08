from ChildProject.projects import ChildProject
from ChildProject.pipelines.pipeline import Pipeline

import pandas as pd


def test_whitelist():
    project = ChildProject("examples/valid_raw_data")
    project.read()

    recordings = project.get_recordings_from_list(
        Pipeline.recordings_from_list(["sound.wav"])
    )
    assert recordings["recording_filename"].tolist() == ["sound.wav"]

    recordings = project.get_recordings_from_list(
        Pipeline.recordings_from_list(pd.Series(["sound.wav"]))
    )
    assert recordings["recording_filename"].tolist() == ["sound.wav"]

    recordings = project.get_recordings_from_list(
        Pipeline.recordings_from_list(
            pd.DataFrame({"recording_filename": ["sound.wav"]})
        )
    )
    assert recordings["recording_filename"].tolist() == ["sound.wav"]

    recordings = pd.DataFrame({"recording_filename": ["sound.wav"]}).to_csv(
        "output/filter.csv"
    )

    recordings = project.get_recordings_from_list(
        Pipeline.recordings_from_list("output/filter.csv")
    )
    assert recordings["recording_filename"].tolist() == ["sound.wav"]

    recordings = pd.DataFrame({"filename": ["sound.wav"]}).to_csv("output/filter.csv")

    caught_value_error = False
    try:
        recordings = project.get_recordings_from_list(
            Pipeline.recordings_from_list("output/filter.csv")
        )
    except ValueError:
        caught_value_error = True

    assert caught_value_error == True

    recordings = project.get_recordings_from_list(
        Pipeline.recordings_from_list(
            [
                "examples/valid_raw_data/recordings/raw/sound.wav",
                "examples/valid_raw_data/recordings/raw/sound2.wav",
            ]
        )
    )
    assert recordings["recording_filename"].tolist() == ["sound.wav", "sound2.wav"]
