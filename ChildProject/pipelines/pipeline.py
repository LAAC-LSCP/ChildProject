from abc import ABC, abstractmethod

class Pipeline(ABC):
    def __init__(self):
        pass

    def check_setup(self):
        pass

    def setup(self):
        pass

    @abstractmethod
    def run(self, **kwargs):
        pass

    @staticmethod
    def setup_pipeline(parser):
        pass

    @staticmethod
    def recordings_from_list(recordings):
        import pandas as pd
        from os.path import exists

        if recordings is None:
            df = None
        elif isinstance(recordings, pd.DataFrame):
            if "recording_filename" not in recordings.columns:
                raise ValueError(
                    "recordings dataframe is missing a 'recording_filename' column"
                )
            df = recordings["recording_filename"].tolist()
        elif isinstance(recordings, pd.Series):
            df = recordings.tolist()
        elif isinstance(recordings, list):
            df = recordings
        else:
            if not exists(recordings):
                raise ValueError(
                    "'recordings' is neither a pandas dataframe,"
                    "nor a list or a path to an existing dataframe."
                )

            df = pd.read_csv(recordings)
            if "recording_filename" not in df.columns:
                raise ValueError(
                    f"'{recordings}' is missing a 'recording_filename' column"
                )
            df = df["recording_filename"].tolist()

        if df is not None:
            df = list(set(df))

        return df