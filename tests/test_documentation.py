from ChildProject.projects import ChildProject
import pandas as pd


def standardize_dataframe(df, columns):
    df = df[list(columns)]
    return df.sort_index(axis=1).sort_values(list(columns)).reset_index(drop=True)


def test_read():
    project = ChildProject("examples/valid_raw_data")
    project.read()

    doc = project.read_documentation()
    truth = pd.read_csv("tests/truth/docs.csv")

    pd.testing.assert_frame_equal(
        standardize_dataframe(doc, columns=truth.columns),
        standardize_dataframe(truth, columns=truth.columns),
    )