import os
import re
import datetime
from typing import Union, Set, List, Tuple

import numpy as np
import pandas as pd

from pydantic import ValidationError



class MissingColumnsException(Exception):
    def __init__(self, name: str, missing: Set):
        missing = ",".join(list(missing))

        super().__init__(
            f"dataframe {name} misses the following required columns: {missing}"
        )
        
class IncorrectDtypeException(Exception):
    """Exception when an Unexpected DType is found in a pandas DataFrame
    """

def assert_dataframe(name: str, df: pd.DataFrame, not_empty: bool = False):
    assert isinstance(
        df, pd.DataFrame
    ), f"{name} should be a dataframe, but type is '{type(df)}' instead."

    if not_empty:
        assert len(df) > 0, f"{name} should not be empty."


def assert_columns_presence(name: str, df: pd.DataFrame, columns: Union[Set, List]):
    missing = set(columns) - set(df.columns)

    if len(missing):
        raise MissingColumnsException(name, missing)
        
def read_csv_with_dtype(file: str, dtypes: dict) -> pd.DataFrame:
    try:
        df = pd.read_csv(file, dtype=dtypes, dtype_backend='numpy_nullable')
    except ValueError:
        raise IncorrectDtypeException('Incorrect type found in {}, expected column types are:\n{}'.format(file, dtypes))
    return df


def is_boolean(x):
    return x == "NA" or int(x) in [0, 1]


class IndexColumn:
    def __init__(
        self,
        name="",
        description="",
        required=False,
        regex=None,
        filename=False,
        directory=None,
        datetime=None,
        function=None,
        choices=None,
        dtype=None,
        unique=False,
        generated=False,
        annotation_columns=None,
        vfield=None,
    ):
        self.name = name
        self.description = description
        self.required = required
        self.filename = filename
        self.directory = directory
        self.regex = regex
        self.datetime = datetime
        self.function = function
        self.choices = choices
        self.unique = unique
        self.generated = generated
        self.dtype = dtype
        self.annotation_columns = annotation_columns
        self.vfield = vfield

    def __str__(self):
        return "IndexColumn(name = {})".format(self.name)

    def __repr__(self):
        return "IndexColumn(name = {})".format(self.name)


class IndexTable:
    def __init__(self, name, path=None, columns=[], enforce_dtypes: bool = False, validator=None):
        self.name = name
        self.path = path
        self.columns = columns
        self.df = None
        self.enforce_dtypes = enforce_dtypes
        self.validator = validator

    def read(self) -> pd.DataFrame:
        pd_flags = {
            "keep_default_na": False,
            "na_values": [
                "-1.#IND",
                "1.#QNAN",
                "1.#IND",
                "-1.#QNAN",
                "#N/A N/A",
                "#N/A",
                "N/A",
                "n/a",
                "",
                "#NA",
                "NULL",
                "null",
                "NaN",
                "-NaN",
                "nan",
                "-nan",
                "",
            ],
            "parse_dates": False,
            "index_col": False,
        }

        if self.enforce_dtypes:
            dtype = {
                column.name: column.dtype for column in self.columns if column.dtype
            }
            self.df = pd.read_csv(self.path, dtype=dtype, **pd_flags, dtype_backend='numpy_nullable')
        else:
            self.df = pd.read_csv(self.path, **pd_flags, dtype_backend='numpy_nullable')

        self.df.index = self.df.index + 2
        return self.df

    def msg(self, text) -> str:
        return "{}: {}".format(self.path, text)

    def validate(self) -> Tuple[List[str], List[str]]:
        errors, warnings = [], []

        columns = {c.name: c for c in self.columns}

        if self.validator is not None:
            try:
                self.validator.validate(self.df)
            except ValidationError as e:
                errors.append(
                    self.msg("\n{}".format(e))
                )

        uniques = [c.name for c in self.columns if c.unique]

        for unique in uniques:
            duplicates = self.df[unique][self.df.duplicated(subset=[unique], keep=False)]
            if duplicates.shape[0]:
                errors.append(
                    self.msg(
                        "Duplicated values when it should be unique for column {}, values {} on lines {} appear multiple times".format(
                            unique,
                            set(duplicates.values),
                            set(duplicates.index),
                        )
                    )
                )

        unknown_columns = [c for c in self.df.columns if c not in columns.keys()]

        if len(unknown_columns) > 0:
            warnings.append(
                self.msg(
                    "unknown column{} '{}' in {}, expected columns are: {}".format(
                        "s" if len(unknown_columns) > 1 else "",
                        ",".join(unknown_columns),
                        self.name,
                        ",".join(columns.keys()),
                    )
                )
            )



        return errors, warnings
